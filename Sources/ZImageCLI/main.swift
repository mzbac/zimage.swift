import Foundation
import Dispatch
import Logging
import Metal
import MLX
import ZImage

LoggingSystem.bootstrap { label in
  var handler = StreamLogHandler.standardError(label: label)
  handler.logLevel = .info
  return handler
}

struct ZImageCLI {
  static var logger: Logger = {
    var logger = Logger(label: "z-image.cli")
    logger.logLevel = .info
    return logger
  }()

  static func run() throws {
    if let dev = MTLCreateSystemDefaultDevice() {
      logger.info("Metal device: \(dev.name)")
    } else {
      logger.warning("No Metal device detected; MLX will fall back to CPU.")
    }

    var prompt: String?
    var negativePrompt: String?
    var width = ZImageModelMetadata.recommendedWidth
    var height = ZImageModelMetadata.recommendedHeight
    var steps = ZImageModelMetadata.recommendedInferenceSteps
    var guidance = ZImageModelMetadata.recommendedGuidanceScale
    var seed: UInt64?
    var outputPath = "z-image.png"
    var model: String?
    var cacheLimit: Int?
    var maxSequenceLength = 512
    var loraPath: String?
    var loraScale: Float = 1.0

    let args = Array(CommandLine.arguments.dropFirst())
    var iterator = args.makeIterator()

    while let arg = iterator.next() {
      switch arg {
      case "--prompt", "-p":
        prompt = nextValue(for: arg, iterator: &iterator)
      case "--negative-prompt", "--np":
        negativePrompt = nextValue(for: arg, iterator: &iterator)
      case "--width", "-W":
        width = intValue(for: arg, iterator: &iterator, minimum: 64, fallback: width)
      case "--height", "-H":
        height = intValue(for: arg, iterator: &iterator, minimum: 64, fallback: height)
      case "--steps", "-s":
        steps = intValue(for: arg, iterator: &iterator, minimum: 1, fallback: steps)
      case "--guidance", "-g":
        guidance = floatValue(for: arg, iterator: &iterator, fallback: guidance)
      case "--seed":
        seed = UInt64(nextValue(for: arg, iterator: &iterator))
      case "--output", "-o":
        outputPath = nextValue(for: arg, iterator: &iterator)
      case "--model", "-m":
        model = nextValue(for: arg, iterator: &iterator)
      case "--cache-limit":
        cacheLimit = intValue(for: arg, iterator: &iterator, minimum: 1, fallback: 2048)
      case "--max-sequence-length":
        maxSequenceLength = intValue(for: arg, iterator: &iterator, minimum: 64, fallback: 512)
      case "--lora", "-l":
        loraPath = nextValue(for: arg, iterator: &iterator)
      case "--lora-scale":
        loraScale = floatValue(for: arg, iterator: &iterator, fallback: 1.0)
      case "--help", "-h":
        printUsage()
        return
      case "quantize":
        try runQuantize(args: Array(args.dropFirst()))
        return
      default:
        logger.warning("Unknown argument: \(arg)")
      }
    }

    guard let prompt else {
      printUsage()
      return
    }

    if let limit = cacheLimit {
      GPU.set(cacheLimit: limit * 1024 * 1024)
      logger.info("GPU cache limit set to \(limit)MB")
    }

    let request = ZImageGenerationRequest(
      prompt: prompt,
      negativePrompt: negativePrompt,
      width: width,
      height: height,
      steps: steps,
      guidanceScale: guidance,
      seed: seed,
      outputPath: URL(fileURLWithPath: outputPath),
      model: model,
      maxSequenceLength: maxSequenceLength,
      loraPath: loraPath,
      loraScale: loraScale
    )

    let pipeline = ZImagePipeline(logger: logger)
    let semaphore = DispatchSemaphore(value: 0)
    Task {
      do {
        _ = try await pipeline.generate(request)
      } catch {
        logger.error("Generation failed: \(error)")
      }
      semaphore.signal()
    }
    semaphore.wait()
  }

  private static func printUsage() {
    print("""
    Z-Image-Turbo Swift port

    Usage: ZImageCLI --prompt "text" [options]
      --prompt, -p           Text prompt (required)
      --negative-prompt      Negative prompt
      --width, -W            Output width (default \(ZImageModelMetadata.recommendedWidth))
      --height, -H           Output height (default \(ZImageModelMetadata.recommendedHeight))
      --steps, -s            Inference steps (default \(ZImageModelMetadata.recommendedInferenceSteps))
      --guidance, -g         Guidance scale (default \(ZImageModelMetadata.recommendedGuidanceScale))
      --seed                 Random seed
      --output, -o           Output path (default z-image.png)
      --model, -m            Model path or HuggingFace ID (default: \(ZImageRepository.id))
      --cache-limit          GPU memory cache limit in MB (default: unlimited)
      --max-sequence-length  Maximum sequence length for text encoding (default: 512)
      --lora, -l             LoRA weights path or HuggingFace ID
      --lora-scale           LoRA scale factor (default: 1.0)
      --help, -h             Show help

    Subcommands:
      quantize               Quantize model weights
        --input, -i          Input model directory (required)
        --output, -o         Output directory (required)
        --bits               Bit width: 4 or 8 (default: 8)
        --group-size         Group size: 32, 64, 128 (default: 32)
        --verbose            Show progress

    Examples:
      ZImageCLI -p "a cute cat" -o cat.png
      ZImageCLI -p "a sunset" -m models/z-image-turbo-q8
      ZImageCLI -p "a forest" -m Tongyi-MAI/Z-Image-Turbo
      ZImageCLI -p "a cut a cat" --lora ostris/z_image_turbo_childrens_drawings
    """)
  }

  private static func runQuantize(args: [String]) throws {
    var input: String?
    var output: String?
    var bits = 8
    var groupSize = 32
    var verbose = false

    var iterator = args.makeIterator()
    while let arg = iterator.next() {
      switch arg {
      case "--input", "-i":
        input = nextValue(for: arg, iterator: &iterator)
      case "--output", "-o":
        output = nextValue(for: arg, iterator: &iterator)
      case "--bits":
        bits = intValue(for: arg, iterator: &iterator, minimum: 1, fallback: bits)
      case "--group-size":
        groupSize = intValue(for: arg, iterator: &iterator, minimum: 1, fallback: groupSize)
      case "--verbose":
        verbose = true
      case "--help", "-h":
        printQuantizeUsage()
        return
      default:
        logger.warning("Unknown quantize argument: \(arg)")
      }
    }

    guard let inputPath = input else {
      logger.error("Missing required --input argument")
      printQuantizeUsage()
      return
    }

    guard let outputPath = output else {
      logger.error("Missing required --output argument")
      printQuantizeUsage()
      return
    }

    let inputURL = URL(fileURLWithPath: inputPath)
    let outputURL = URL(fileURLWithPath: outputPath)

    guard FileManager.default.fileExists(atPath: inputURL.path) else {
      logger.error("Input directory not found: \(inputPath)")
      return
    }

    guard ZImageQuantizer.supportedBits.contains(bits) else {
      logger.error("Invalid bits: \(bits). Supported: 4, 8")
      return
    }

    guard ZImageQuantizer.supportedGroupSizes.contains(groupSize) else {
      logger.error("Invalid group size: \(groupSize). Supported: 32, 64, 128")
      return
    }

    let spec = ZImageQuantizationSpec(groupSize: groupSize, bits: bits, mode: .affine)

    print("Quantizing: \(inputPath) -> \(outputPath)")
    print("Config: \(bits)-bit, group_size=\(groupSize)")

    try ZImageQuantizer.quantizeAndSave(
      from: inputURL,
      to: outputURL,
      spec: spec,
      verbose: verbose
    )

    print("Done: \(outputURL.path)")
  }

  private static func printQuantizeUsage() {
    print("""
    Quantize model weights.

    Usage: ZImageCLI quantize -i <input> -o <output> [options]
      --input, -i          Input model directory (required)
      --output, -o         Output directory (required)
      --bits               Bit width: 4 or 8 (default: 8)
      --group-size         Group size: 32, 64, 128 (default: 32)
      --verbose            Show progress
      --help, -h           Show help

    Example:
      ZImageCLI quantize -i models/z-image-turbo -o models/z-image-turbo-q8 --verbose
    """)
  }

  private static func nextValue(for arg: String, iterator: inout IndexingIterator<[String]>) -> String {
    guard let value = iterator.next() else {
      fatalError("Expected value after \(arg)")
    }
    return value
  }

  private static func intValue(for arg: String, iterator: inout IndexingIterator<[String]>, minimum: Int, fallback: Int) -> Int {
    guard let value = Int(nextValue(for: arg, iterator: &iterator)) else { return fallback }
    return max(minimum, value)
  }

  private static func floatValue(for arg: String, iterator: inout IndexingIterator<[String]>, fallback: Float) -> Float {
    Float(nextValue(for: arg, iterator: &iterator)) ?? fallback
  }
}

try ZImageCLI.run()
