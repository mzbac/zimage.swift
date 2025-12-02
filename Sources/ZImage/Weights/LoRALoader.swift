import Foundation
import Hub
import Logging
import MLX
import MLXNN

public struct LoRALoader {
  private let logger: Logger
  private let hubApi: HubApi

  public init(logger: Logger = Logger(label: "z-image.lora"), hubApi: HubApi = .shared) {
    self.logger = logger
    self.hubApi = hubApi
  }

  public func loadLoRAWeights(from loraPath: String, dtype: DType = .bfloat16) async throws -> [String: MLXArray] {
    let loraDirectory: URL

    if FileManager.default.fileExists(atPath: loraPath) {
      loraDirectory = URL(fileURLWithPath: loraPath)
      logger.info("Loading LoRA from local path: \(loraPath)")
    } else {
      logger.info("Downloading LoRA from HuggingFace: \(loraPath)")
      let repo = Hub.Repo(id: loraPath)
      loraDirectory = try await hubApi.snapshot(
        from: repo,
        matching: ["*.safetensors"]
      ) { progress in
        let percent = Int(progress.fractionCompleted * 100)
        if percent % 20 == 0 {
          self.logger.info("LoRA download: \(percent)%")
        }
      }
    }

    return try Self.loadLoRAWeights(directory: loraDirectory, dtype: dtype, logger: logger)
  }

  public static func loadLoRAWeights(directory: URL, dtype: DType, logger: Logger) throws -> [String: MLXArray] {
    var loraWeights = [String: MLXArray]()

    guard let enumerator = FileManager.default.enumerator(
      at: directory, includingPropertiesForKeys: nil
    ) else {
      throw LoRAError.directoryNotFound(directory.path)
    }

    for case let url as URL in enumerator {
      if url.pathExtension == "safetensors" {
        logger.info("Loading LoRA weights from: \(url.lastPathComponent)")
        let weights = try MLX.loadArrays(url: url)
        for (key, value) in weights {
          let newKey = remapWeightKey(key)
          if value.dtype != dtype {
            loraWeights[newKey] = value.asType(dtype)
          } else {
            loraWeights[newKey] = value
          }
        }
      }
    }

    logger.info("Loaded \(loraWeights.count) LoRA tensors")
    return loraWeights
  }

  internal static func remapWeightKey(_ key: String) -> String {
    var newKey = key

    // Handle "lora_unet_" prefix common in Diffusers/PEFT format
    if newKey.hasPrefix("lora_unet_") {
      newKey = String(newKey.dropFirst("lora_unet_".count))
    }

    // Handle "diffusion_model." prefix common in Z-Image LoRA format
    if newKey.hasPrefix("diffusion_model.") {
      newKey = String(newKey.dropFirst("diffusion_model.".count))
    }

    // Handle ".ff." or ".ff_context." for feed-forward layers (Flux format)
    if newKey.contains(".ff.") || newKey.contains(".ff_context.") {
      let components = newKey.components(separatedBy: ".")
      if components.count >= 5 {
        let blockIndex = components[1]
        let ffType = components[2]
        let netIndex = components[4]

        if netIndex == "0" {
          return "transformer_blocks.\(blockIndex).\(ffType).linear1.\(components.last ?? "")"
        } else if netIndex == "2" {
          return "transformer_blocks.\(blockIndex).\(ffType).linear2.\(components.last ?? "")"
        }
      }
    }

    return newKey
  }
}

public func applyLoRAWeights(
  to transformer: ZImageTransformer2DModel,
  loraWeights: [String: MLXArray],
  loraScale: Float = 1.0,
  logger: Logger
) {
  var layerUpdates: [String: MLXArray] = [:]
  var appliedCount = 0

  for (key, module) in transformer.namedModules() {
    // Try different key patterns for LoRA weights
    // The LoRA keys after remapping should match the transformer module keys
    let keyPatterns = [
      key,
      "transformer.\(key)",
      key.replacingOccurrences(of: ".", with: "_")
    ]

    for pattern in keyPatterns {
      let loraAKey = "\(pattern).lora_A.weight"
      let loraBKey = "\(pattern).lora_B.weight"
      let loraAKeyAlt = "\(pattern).lora_down.weight"
      let loraBKeyAlt = "\(pattern).lora_up.weight"

      let loraA = loraWeights[loraAKey] ?? loraWeights[loraAKeyAlt]
      let loraB = loraWeights[loraBKey] ?? loraWeights[loraBKeyAlt]

      if let loraA = loraA, let loraB = loraB {
        if let quantizedLinear = module as? QuantizedLinear {
          logger.debug("Applying LoRA to quantized layer: \(key)")

          let dequantizedWeight = dequantized(
            quantizedLinear.weight,
            scales: quantizedLinear.scales,
            biases: quantizedLinear.biases,
            groupSize: quantizedLinear.groupSize,
            bits: quantizedLinear.bits
          )

          let loraDelta = matmul(loraB, loraA)
          let fusedWeight = dequantizedWeight + loraScale * loraDelta

          let fusedLinear = Linear(
            weight: fusedWeight,
            bias: quantizedLinear.bias
          )

          let requantized = QuantizedLinear(
            fusedLinear,
            groupSize: quantizedLinear.groupSize,
            bits: quantizedLinear.bits
          )

          layerUpdates["\(key).weight"] = requantized.weight
          layerUpdates["\(key).scales"] = requantized.scales
          layerUpdates["\(key).biases"] = requantized.biases
          appliedCount += 1

        } else if let linear = module as? Linear {
          logger.debug("Applying LoRA to linear layer: \(key)")
          let loraDelta = matmul(loraB, loraA)
          let currentWeight = linear.weight
          let newWeight = currentWeight + loraScale * loraDelta
          layerUpdates["\(key).weight"] = newWeight
          appliedCount += 1
        }

        break
      }
    }
  }

  if !layerUpdates.isEmpty {
    do {
      try transformer.update(parameters: ModuleParameters.unflattened(layerUpdates), verify: [.shapeMismatch])
      logger.info("Applied LoRA weights to \(appliedCount) layers")
    } catch {
      logger.error("Failed to apply LoRA weights: \(error)")
    }
  } else {
    logger.warning("No matching LoRA weights found for transformer layers")
  }
}

public enum LoRAError: Error, LocalizedError {
  case directoryNotFound(String)
  case weightsNotFound(String)
  case applicationFailed(String)

  public var errorDescription: String? {
    switch self {
    case .directoryNotFound(let path):
      return "LoRA directory not found: \(path)"
    case .weightsNotFound(let path):
      return "LoRA weights not found at: \(path)"
    case .applicationFailed(let reason):
      return "Failed to apply LoRA: \(reason)"
    }
  }
}
