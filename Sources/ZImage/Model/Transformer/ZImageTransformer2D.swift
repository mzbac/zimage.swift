import Foundation
import MLX
import MLXNN

private struct TransformerCacheKey: Hashable {
  let batch: Int
  let height: Int
  let width: Int
  let frames: Int
  let capOriLen: Int
}

private struct TransformerCache {
  let capFreqs: MLXArray
  let capPadMask: MLXArray?
  let capSeqLen: Int
  let capPad: Int

  let imgFreqs: MLXArray
  let imgPadMask: MLXArray?
  let imgSeqLen: Int
  let imgPad: Int
  let imageTokens: Int

  let unifiedFreqsCis: MLXArray

  let fTokens: Int
  let hTokens: Int
  let wTokens: Int
}

public final class ZImageTransformer2DModel: Module {
  public let configuration: ZImageTransformerConfig
  @ModuleInfo(key: "t_embedder") var tEmbedder: ZImageTimestepEmbedder
  @ModuleInfo(key: "all_x_embedder") var allXEmbedder: [String: Linear]
  @ModuleInfo(key: "all_final_layer") var allFinalLayer: [String: ZImageFinalLayer]
  @ModuleInfo(key: "noise_refiner") public internal(set) var noiseRefiner: [ZImageTransformerBlock]
  @ModuleInfo(key: "context_refiner") public internal(set) var contextRefiner: [ZImageTransformerBlock]
  @ModuleInfo(key: "layers") public internal(set) var layers: [ZImageTransformerBlock]
  var capEmbedNorm: RMSNorm
  var capEmbedLinear: Linear

  let ropeEmbedder: ZImageRopeEmbedder
  private var xPadToken: MLXArray?
  private var capPadToken: MLXArray?

  private var cache: TransformerCache?
  private var cacheKey: TransformerCacheKey?

  public init(configuration: ZImageTransformerConfig) {
    self.configuration = configuration
    let outSize = min(configuration.dim, 256)
    self._tEmbedder.wrappedValue = ZImageTimestepEmbedder(outSize: outSize, midSize: 1024)

    let patchSize = 2
    let fPatchSize = 1
    let key = "\(patchSize)-\(fPatchSize)"
    var xEmbedder: [String: Linear] = [:]
    var finalLayers: [String: ZImageFinalLayer] = [:]

    let inFeatures = fPatchSize * patchSize * patchSize * configuration.inChannels
    xEmbedder[key] = Linear(inFeatures, configuration.dim, bias: true)
    finalLayers[key] = ZImageFinalLayer(
      hiddenSize: configuration.dim,
      outChannels: patchSize * patchSize * fPatchSize * configuration.inChannels
    )
    self._allXEmbedder.wrappedValue = xEmbedder
    self._allFinalLayer.wrappedValue = finalLayers

    self.capEmbedNorm = RMSNorm(dimensions: configuration.capFeatDim, eps: configuration.normEps)
    self.capEmbedLinear = Linear(configuration.capFeatDim, configuration.dim, bias: true)

    var noiseBlocks: [ZImageTransformerBlock] = []
    for layerId in 0..<configuration.nRefinerLayers {
      noiseBlocks.append(
        ZImageTransformerBlock(
          layerId: 1000 + layerId,
          dim: configuration.dim,
          nHeads: configuration.nHeads,
          nKvHeads: configuration.nKVHeads,
          normEps: configuration.normEps,
          qkNorm: configuration.qkNorm,
          modulation: true
        )
      )
    }
    self._noiseRefiner.wrappedValue = noiseBlocks

    var contextBlocks: [ZImageTransformerBlock] = []
    for layerId in 0..<configuration.nRefinerLayers {
      contextBlocks.append(
        ZImageTransformerBlock(
          layerId: layerId,
          dim: configuration.dim,
          nHeads: configuration.nHeads,
          nKvHeads: configuration.nKVHeads,
          normEps: configuration.normEps,
          qkNorm: configuration.qkNorm,
          modulation: false
        )
      )
    }
    self._contextRefiner.wrappedValue = contextBlocks

    var mainLayers: [ZImageTransformerBlock] = []
    for layerId in 0..<configuration.nLayers {
      mainLayers.append(
        ZImageTransformerBlock(
          layerId: layerId,
          dim: configuration.dim,
          nHeads: configuration.nHeads,
          nKvHeads: configuration.nKVHeads,
          normEps: configuration.normEps,
          qkNorm: configuration.qkNorm,
          modulation: true
        )
      )
    }
    self._layers.wrappedValue = mainLayers

    self.ropeEmbedder = ZImageRopeEmbedder(
      theta: configuration.ropeTheta,
      axesDims: configuration.axesDims,
      axesLens: configuration.axesLens
    )
    super.init()
  }

  public func loadCapEmbedderWeights(from weights: [String: MLXArray]) {
    if let normWeight = weights["cap_embedder.0.weight"] {
      capEmbedNorm.weight._updateInternal(normWeight)
    }
    if let linearWeight = weights["cap_embedder.1.weight"] {
      capEmbedLinear.weight._updateInternal(linearWeight)
    }
    if let linearBias = weights["cap_embedder.1.bias"] {
      capEmbedLinear.bias?._updateInternal(linearBias)
    }
  }

  public func loadXEmbedderWeights(from weights: [String: MLXArray], groupSize: Int = 32, bits: Int = 8) {
    let key = "2-1"
    let prefix = "all_x_embedder.\(key)"
    guard let linear = allXEmbedder[key] else { return }

    if let w = weights["\(prefix).weight"] {
      linear.weight._updateInternal(w)
    }
    if let b = weights["\(prefix).bias"] {
      linear.bias?._updateInternal(b)
    }
  }

  public func loadFinalLayerWeights(from weights: [String: MLXArray], groupSize: Int = 32, bits: Int = 8) {
    let key = "2-1"
    let prefix = "all_final_layer.\(key)"
    guard let finalLayer = allFinalLayer[key] else { return }

    if let lin = finalLayer.linear as? Linear {
      if let w = weights["\(prefix).linear.weight"] { lin.weight._updateInternal(w) }
      if let b = weights["\(prefix).linear.bias"] { lin.bias?._updateInternal(b) }
    }

    if let lin = finalLayer.adaLN.linear as? Linear {
      if let w = weights["\(prefix).adaLN_modulation.1.weight"] { lin.weight._updateInternal(w) }
      if let b = weights["\(prefix).adaLN_modulation.1.bias"] { lin.bias?._updateInternal(b) }
    }
  }

  public func setPadTokens(xPad: MLXArray?, capPad: MLXArray?) {
    if let xPad {
      let padDim = xPad.dim(xPad.ndim - 1)
      self.xPadToken = xPad.reshaped(padDim)
    }
    if let capPad {
      let padDim = capPad.dim(capPad.ndim - 1)
      self.capPadToken = capPad.reshaped(padDim)
    }
  }

  public func clearCache() {
    cache = nil
    cacheKey = nil
  }

  private func getOrBuildCache(
    batch: Int,
    height: Int,
    width: Int,
    frames: Int,
    capOriLen: Int,
    patchSize: Int,
    fPatchSize: Int
  ) -> TransformerCache {
    let seqMultiOf = 32
    let key = TransformerCacheKey(
      batch: batch,
      height: height,
      width: width,
      frames: frames,
      capOriLen: capOriLen
    )

    if let existingKey = cacheKey, let existingCache = cache, existingKey == key {
      return existingCache
    }

    let capPad = (seqMultiOf - (capOriLen % seqMultiOf)) % seqMultiOf
    let capSeqLen = capOriLen + capPad

    let capPosIds = ZImageCoordinateUtils.createCoordinateGrid(
      size: (capSeqLen, 1, 1),
      start: (1, 0, 0)
    ).reshaped(capSeqLen, 3)
    let capFreqs = ropeEmbedder(ids: capPosIds)

    var capPadMask: MLXArray? = nil
    if capPad > 0 {
      let capPadMask1d = MLX.concatenated([
        MLX.zeros([capOriLen], dtype: .bool),
        MLX.ones([capPad], dtype: .bool)
      ], axis: 0)
      capPadMask = MLX.broadcast(capPadMask1d.reshaped(1, capSeqLen), to: [batch, capSeqLen])
    }

    let fTokens = frames / fPatchSize
    let hTokens = height / patchSize
    let wTokens = width / patchSize
    let imageTokens = fTokens * hTokens * wTokens
    let imgPad = (seqMultiOf - (imageTokens % seqMultiOf)) % seqMultiOf
    let imgSeqLen = imageTokens + imgPad

    let imgPos = ZImageCoordinateUtils.createCoordinateGrid(
      size: (fTokens, hTokens, wTokens),
      start: (capSeqLen + 1, 0, 0)
    ).reshaped(imageTokens, 3)
    let imgPadPos = ZImageCoordinateUtils.createCoordinateGrid(
      size: (imgPad, 1, 1),
      start: (0, 0, 0)
    ).reshaped(imgPad, 3)
    let imgPosIds = MLX.concatenated([imgPos, imgPadPos], axis: 0)
    let imgFreqs = ropeEmbedder(ids: imgPosIds)

    var imgPadMask: MLXArray? = nil
    if imgPad > 0 {
      let imgPadMask1d = MLX.concatenated([
        MLX.zeros([imageTokens], dtype: .bool),
        MLX.ones([imgPad], dtype: .bool)
      ], axis: 0)
      imgPadMask = MLX.broadcast(imgPadMask1d.reshaped(1, imgSeqLen), to: [batch, imgSeqLen])
    }

    let unifiedFreqsCis = MLX.concatenated([imgFreqs, capFreqs], axis: 0)

    let newCache = TransformerCache(
      capFreqs: capFreqs,
      capPadMask: capPadMask,
      capSeqLen: capSeqLen,
      capPad: capPad,
      imgFreqs: imgFreqs,
      imgPadMask: imgPadMask,
      imgSeqLen: imgSeqLen,
      imgPad: imgPad,
      imageTokens: imageTokens,
      unifiedFreqsCis: unifiedFreqsCis,
      fTokens: fTokens,
      hTokens: hTokens,
      wTokens: wTokens
    )

    self.cache = newCache
    self.cacheKey = key

    return newCache
  }

  public func forward(
    latents: MLXArray,
    timestep: MLXArray,
    promptEmbeds: MLXArray
  ) -> MLXArray {
    let hasFrameDim = latents.ndim == 5
    let batch = latents.dim(0)
    let channels = latents.dim(1)
    let frames = hasFrameDim ? latents.dim(2) : 1
    let height = latents.dim(hasFrameDim ? 3 : 2)
    let width = latents.dim(hasFrameDim ? 4 : 3)

    let patchSize = 2
    let fPatchSize = 1
    let key = "\(patchSize)-\(fPatchSize)"
    guard let xEmbed = allXEmbedder[key], let finalLayer = allFinalLayer[key] else {
      return MLX.zeros(latents.shape, dtype: latents.dtype)
    }

    let capOriLen = promptEmbeds.dim(1)
    let cached = getOrBuildCache(
      batch: batch,
      height: height,
      width: width,
      frames: frames,
      capOriLen: capOriLen,
      patchSize: patchSize,
      fPatchSize: fPatchSize
    )

    var latentsWithFrame = latents
    if !hasFrameDim {
      latentsWithFrame = MLX.expandedDimensions(latents, axis: 2)
    }

    let tScaled = timestep * MLXArray(configuration.tScale)
    var tEmb = tEmbedder(tScaled)

    var capFeat = promptEmbeds
    if cached.capPad > 0 {
      let last = promptEmbeds[0..., capOriLen - 1, 0...]
      let pad = MLX.broadcast(last, to: [batch, cached.capPad, promptEmbeds.dim(2)])
      capFeat = MLX.concatenated([promptEmbeds, pad], axis: 1)
    }
    capFeat = capEmbedLinear(capEmbedNorm(capFeat))

    if let capPadToken, let capPadMask = cached.capPadMask {
      let padDim = capPadToken.dim(capPadToken.ndim - 1)
      let pad = MLX.broadcast(capPadToken.reshaped(1, 1, padDim), to: [batch, cached.capSeqLen, padDim])
      capFeat = MLX.where(MLX.expandedDimensions(capPadMask, axis: 2), pad, capFeat)
    }

    var image = latentsWithFrame
      .reshaped(batch, channels, cached.fTokens, fPatchSize, cached.hTokens, patchSize, cached.wTokens, patchSize)
      .transposed(0, 2, 4, 6, 3, 5, 7, 1)
      .reshaped(batch, cached.imageTokens, patchSize * patchSize * fPatchSize * channels)

    if cached.imgPad > 0 {
      let last = image[0..., cached.imageTokens - 1, 0...]
      let pad = MLX.broadcast(last, to: [batch, cached.imgPad, image.dim(2)])
      image = MLX.concatenated([image, pad], axis: 1)
    }

    image = xEmbed(image)
    tEmb = tEmb.asType(image.dtype)

    if let xPadToken, let imgPadMask = cached.imgPadMask {
      let padDim = xPadToken.dim(xPadToken.ndim - 1)
      let pad = MLX.broadcast(xPadToken.reshaped(1, 1, padDim), to: [batch, cached.imgSeqLen, padDim])
      image = MLX.where(MLX.expandedDimensions(imgPadMask, axis: 2), pad, image)
    }

    var noiseStream = image
    for block in noiseRefiner {
      noiseStream = block(
        noiseStream,
        attnMask: nil,
        freqsCis: cached.imgFreqs,
        adalnInput: tEmb
      )
    }

    var capStream = capFeat
    for block in contextRefiner {
      capStream = block(
        capStream,
        attnMask: nil,
        freqsCis: cached.capFreqs,
        adalnInput: nil
      )
    }

    var unified = MLX.concatenated([noiseStream, capStream], axis: 1)

    for block in layers {
      unified = block(unified, attnMask: nil, freqsCis: cached.unifiedFreqsCis, adalnInput: tEmb)
    }

    let imageOut = unified[0..., 0..<cached.imageTokens, 0...]
    let projected = finalLayer(imageOut, conditioning: tEmb)
    let outChannels = configuration.inChannels

    var reshaped = projected
      .reshaped(batch, cached.fTokens, cached.hTokens, cached.wTokens, fPatchSize, patchSize, patchSize, outChannels)
      .transposed(0, 7, 1, 4, 2, 5, 3, 6)
      .reshaped(batch, outChannels, cached.fTokens * fPatchSize, cached.hTokens * patchSize, cached.wTokens * patchSize)

    reshaped = reshaped[0..., 0..., 0, 0..., 0...]
    return reshaped
  }
}
