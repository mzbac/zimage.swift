import Foundation
import MLX
import MLXNN
import MLXFast

final class QwenVisionAttention: Module {
  let embedDim: Int
  let numHeads: Int
  let headDim: Int
  let scale: Float

  @ModuleInfo(key: "qkv") private var qkv: Linear
  @ModuleInfo(key: "proj") private var proj: Linear

  init(embedDim: Int, numHeads: Int) {
    precondition(embedDim % numHeads == 0, "embedDim must be divisible by numHeads")
    self.embedDim = embedDim
    self.numHeads = numHeads
    self.headDim = embedDim / numHeads
    self.scale = 1.0 / Float(sqrt(Double(headDim)))

    self._qkv.wrappedValue = Linear(embedDim, embedDim * 3)
    self._proj.wrappedValue = Linear(embedDim, embedDim)
  }

  func callAsFunction(
    _ hiddenStates: MLXArray,
    rotaryEmbedding: (cos: MLXArray, sin: MLXArray)? = nil,
    attentionMask: MLXArray? = nil,
    cuSeqlens: MLXArray? = nil
  ) -> MLXArray {
    var states = hiddenStates

    let batch = states.dim(0)
    let sequence = states.dim(1)
    let baseMask: MLXArray? = attentionMask.flatMap { prepareAttentionMask($0, batch: batch) }

    states = qkv(states)
    states = states.reshaped(batch, sequence, 3, numHeads, headDim)
    // Linear outputs before rotary, flattened over heads
    let qPre = states[0..., 0..., 0..<1, 0..., 0...].squeezed(axis: 2) // [B,S,H,D]
    let kPre = states[0..., 0..., 1..<2, 0..., 0...].squeezed(axis: 2) // [B,S,H,D]
    let vPre = states[0..., 0..., 2..<3, 0..., 0...].squeezed(axis: 2) // [B,S,H,D]

    var q = qPre.transposed(0, 2, 1, 3)
    var k = kPre.transposed(0, 2, 1, 3)
    var v = vPre.transposed(0, 2, 1, 3)

    if let rotaryEmbedding {
      q = applyRotary(q, cos: rotaryEmbedding.cos, sin: rotaryEmbedding.sin)
      k = applyRotary(k, cos: rotaryEmbedding.cos, sin: rotaryEmbedding.sin)
    }

    let maskMode: MLXFast.ScaledDotProductAttentionMaskMode = baseMask.map { .array($0) } ?? .none

    var context = MLXFast.scaledDotProductAttention(
      queries: q,
      keys: k,
      values: v,
      scale: scale,
      mask: maskMode
    )
    context = context.transposed(0, 2, 1, 3)

    context = context.reshaped(batch, sequence, embedDim)
    context = proj(context)
    return context 
  }

  private func prepareAttentionMask(
    _ mask: MLXArray,
    batch: Int
  ) -> MLXArray {
    var prepared = mask
    if prepared.ndim == 2 {
      prepared = prepared[.newAxis, 0..., 0...]
    }
    precondition(
      prepared.ndim == 3,
      "Vision attention mask must have shape [batch, sequence, sequence]"
    )
    precondition(
      prepared.dim(0) == batch || prepared.dim(0) == 1,
      "Vision attention mask batch dimension mismatch"
    )
    if prepared.dim(0) == 1 && batch > 1 {
      let broadcastShape = [batch, prepared.dim(1), prepared.dim(2)]
      let floatMask = prepared.asType(prepared.dtype)
      prepared = MLX.broadcast(floatMask, to: broadcastShape)
    }
    return prepared.asType(.bool)
  }

  private func applyRotary(_ tensor: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
    let cosPrepared = cos[.newAxis, .newAxis, 0..., 0...]
    let sinPrepared = sin[.newAxis, .newAxis, 0..., 0...]
    return (tensor * cosPrepared) + (rotateHalf(tensor) * sinPrepared)
  }

  private func rotateHalf(_ tensor: MLXArray) -> MLXArray {
    let half = tensor.dim(-1) / 2
    let firstHalf = tensor[0..., 0..., 0..., 0..<half]
    let secondHalf = tensor[0..., 0..., 0..., half...]
    return MLX.concatenated([-secondHalf, firstHalf], axis: -1)
  }
}
