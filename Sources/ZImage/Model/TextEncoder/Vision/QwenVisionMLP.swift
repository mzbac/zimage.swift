import Foundation
import MLX
import MLXNN

final class QwenVisionMLP: Module {
  @ModuleInfo(key: "gate") private var gate: Linear
  @ModuleInfo(key: "up") private var up: Linear
  @ModuleInfo(key: "down") private var down: Linear

  private let activation: QwenVisionConfiguration.Activation

  init(dim: Int, hiddenDim: Int, activation: QwenVisionConfiguration.Activation) {
    self.activation = activation
    self._gate.wrappedValue = Linear(dim, hiddenDim)
    self._up.wrappedValue = Linear(dim, hiddenDim)
    self._down.wrappedValue = Linear(hiddenDim, dim)
  }

  func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
    var gated = gate(hiddenStates)
    switch activation {
    case .geluApproximate:
      gated = MLXNN.geluFastApproximate(gated)
    case .silu:
      gated = MLXNN.silu(gated)
    }
    return down(gated * up(hiddenStates))
  }

}
