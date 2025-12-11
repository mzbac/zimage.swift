import XCTest
import MLX
@testable import ZImage

/// Unit tests for prompt enhancement components
final class PromptEnhancementTests: XCTestCase {

  // MARK: - PromptEnhanceConfig Tests

  func testPromptEnhanceConfigDefaultValues() {
    let config = PromptEnhanceConfig()

    XCTAssertEqual(config.maxNewTokens, 512)
    XCTAssertEqual(config.temperature, 0.7)
    XCTAssertEqual(config.topP, 0.9)
    XCTAssertEqual(config.repetitionPenalty, 1.05)
    XCTAssertEqual(config.repetitionContextSize, 20)
    XCTAssertEqual(config.eosTokenId, 151645)
    XCTAssertEqual(config.stopTokenIds, [151645, 151643])
  }

  func testPromptEnhanceConfigCustomValues() {
    let config = PromptEnhanceConfig(
      maxNewTokens: 256,
      temperature: 0.5,
      topP: 0.8,
      repetitionPenalty: 1.1,
      repetitionContextSize: 30,
      eosTokenId: 12345,
      stopTokenIds: [12345, 67890]
    )

    XCTAssertEqual(config.maxNewTokens, 256)
    XCTAssertEqual(config.temperature, 0.5)
    XCTAssertEqual(config.topP, 0.8)
    XCTAssertEqual(config.repetitionPenalty, 1.1)
    XCTAssertEqual(config.repetitionContextSize, 30)
    XCTAssertEqual(config.eosTokenId, 12345)
    XCTAssertEqual(config.stopTokenIds, [12345, 67890])
  }

  // MARK: - GenerationConfig Tests

  func testGenerationConfigDefaultValues() {
    let config = GenerationConfig()

    XCTAssertEqual(config.maxTokens, 256)
    XCTAssertEqual(config.temperature, 0.7)
    XCTAssertEqual(config.topP, 0.9)
    XCTAssertEqual(config.repetitionPenalty, 1.05)
    XCTAssertEqual(config.repetitionContextSize, 20)
  }

  func testGenerationConfigCustomValues() {
    let config = GenerationConfig(
      maxTokens: 512,
      temperature: 0.3,
      topP: 0.95,
      repetitionPenalty: 1.2,
      repetitionContextSize: 50
    )

    XCTAssertEqual(config.maxTokens, 512)
    XCTAssertEqual(config.temperature, 0.3)
    XCTAssertEqual(config.topP, 0.95)
    XCTAssertEqual(config.repetitionPenalty, 1.2)
    XCTAssertEqual(config.repetitionContextSize, 50)
  }

  // MARK: - Sampling Function Tests

  func testArgMaxSample() {
    // Create logits where index 3 has the highest value
    let logits = MLXArray([Float(0.1), Float(0.2), Float(0.5), Float(0.9), Float(0.3)])
    MLX.eval(logits)

    let result = argMaxSample(logits: logits)
    XCTAssertEqual(result, 3)
  }

  func testArgMaxSampleWithNegativeValues() {
    // Test with negative values
    let logits = MLXArray([Float(-0.5), Float(-0.1), Float(-0.3), Float(-0.8)])
    MLX.eval(logits)

    let result = argMaxSample(logits: logits)
    XCTAssertEqual(result, 1) // -0.1 is the highest
  }

  func testApplyRepetitionPenalty() {
    let logits = MLXArray([Float(1.0), Float(2.0), Float(3.0), Float(4.0), Float(5.0)])
    let tokens = [1, 3] // Penalize indices 1 and 3
    let penalty: Float = 2.0

    let result = applyRepetitionPenalty(logits: logits, tokens: tokens, penalty: penalty)
    MLX.eval(result)

    let resultArray = result.asArray(Float.self)

    // Index 0: unchanged (1.0)
    XCTAssertEqual(resultArray[0], 1.0, accuracy: 1e-5)
    // Index 1: positive value divided by penalty (2.0 / 2.0 = 1.0)
    XCTAssertEqual(resultArray[1], 1.0, accuracy: 1e-5)
    // Index 2: unchanged (3.0)
    XCTAssertEqual(resultArray[2], 3.0, accuracy: 1e-5)
    // Index 3: positive value divided by penalty (4.0 / 2.0 = 2.0)
    XCTAssertEqual(resultArray[3], 2.0, accuracy: 1e-5)
    // Index 4: unchanged (5.0)
    XCTAssertEqual(resultArray[4], 5.0, accuracy: 1e-5)
  }

  func testApplyRepetitionPenaltyWithNegativeLogits() {
    let logits = MLXArray([Float(-2.0), Float(-1.0), Float(0.0), Float(1.0), Float(2.0)])
    let tokens = [0, 4] // Penalize indices 0 and 4
    let penalty: Float = 2.0

    let result = applyRepetitionPenalty(logits: logits, tokens: tokens, penalty: penalty)
    MLX.eval(result)

    let resultArray = result.asArray(Float.self)

    // Index 0: negative value multiplied by penalty (-2.0 * 2.0 = -4.0)
    XCTAssertEqual(resultArray[0], -4.0, accuracy: 1e-5)
    // Index 4: positive value divided by penalty (2.0 / 2.0 = 1.0)
    XCTAssertEqual(resultArray[4], 1.0, accuracy: 1e-5)
  }

  func testApplyRepetitionPenaltyNoOpWithPenaltyOne() {
    let logits = MLXArray([Float(1.0), Float(2.0), Float(3.0)])
    let tokens = [0, 1, 2]
    let penalty: Float = 1.0

    let result = applyRepetitionPenalty(logits: logits, tokens: tokens, penalty: penalty)
    MLX.eval(result)

    let resultArray = result.asArray(Float.self)
    XCTAssertEqual(resultArray[0], 1.0, accuracy: 1e-5)
    XCTAssertEqual(resultArray[1], 2.0, accuracy: 1e-5)
    XCTAssertEqual(resultArray[2], 3.0, accuracy: 1e-5)
  }

  func testApplyRepetitionPenaltyNoOpWithEmptyTokens() {
    let logits = MLXArray([Float(1.0), Float(2.0), Float(3.0)])
    let tokens: [Int] = []
    let penalty: Float = 2.0

    let result = applyRepetitionPenalty(logits: logits, tokens: tokens, penalty: penalty)
    MLX.eval(result)

    let resultArray = result.asArray(Float.self)
    XCTAssertEqual(resultArray[0], 1.0, accuracy: 1e-5)
    XCTAssertEqual(resultArray[1], 2.0, accuracy: 1e-5)
    XCTAssertEqual(resultArray[2], 3.0, accuracy: 1e-5)
  }

  func testSampleTokenWithZeroTemperature() {
    // With temperature=0, should use argmax (deterministic)
    let logits = MLXArray([Float(0.1), Float(0.5), Float(0.9), Float(0.3)])
    MLX.eval(logits)

    let config = GenerationConfig(
      temperature: 0,
      topP: 0.9,
      repetitionPenalty: nil
    )

    let result = sampleToken(logits: logits, config: config, previousTokens: [])
    XCTAssertEqual(result, 2) // Index 2 has highest value (0.9)
  }

  func testSampleTokenWithZeroTemperatureAndRepetitionPenalty() {
    // Test that repetition penalty is applied before argmax
    let logits = MLXArray([Float(0.1), Float(0.5), Float(0.9), Float(0.3)])
    MLX.eval(logits)

    let config = GenerationConfig(
      temperature: 0,
      topP: 0.9,
      repetitionPenalty: 10.0, // Heavy penalty
      repetitionContextSize: 20
    )

    // Penalize index 2 (the highest)
    let result = sampleToken(logits: logits, config: config, previousTokens: [2])
    // After heavy penalty on index 2, index 1 (0.5) should be highest
    XCTAssertEqual(result, 1)
  }

  func testTopPSampleReturnsBoundedResult() {
    // Test that top-p sampling returns a valid index
    let logits = MLXArray([Float(0.1), Float(0.2), Float(0.3), Float(0.4)])
    MLX.eval(logits)

    let result = topPSample(logits: logits, temperature: 0.7, topP: 0.9)

    XCTAssertGreaterThanOrEqual(result, 0)
    XCTAssertLessThan(result, 4)
  }

  func testCategoricalSampleReturnsBoundedResult() {
    // Test that categorical sampling returns a valid index
    let logits = MLXArray([Float(0.1), Float(0.2), Float(0.3), Float(0.4)])
    MLX.eval(logits)

    let result = categoricalSample(logits: logits, temperature: 0.7)

    XCTAssertGreaterThanOrEqual(result, 0)
    XCTAssertLessThan(result, 4)
  }

  func testSampleTokenWithTopP() {
    // Test that top-p path is taken when 0 < topP < 1
    let logits = MLXArray([Float(0.1), Float(0.2), Float(0.3), Float(0.4)])
    MLX.eval(logits)

    let config = GenerationConfig(
      temperature: 0.7,
      topP: 0.5, // 0 < topP < 1 triggers top-p sampling
      repetitionPenalty: nil
    )

    // Run multiple times to verify it returns valid indices
    for _ in 0..<10 {
      let result = sampleToken(logits: logits, config: config, previousTokens: [])
      XCTAssertGreaterThanOrEqual(result, 0)
      XCTAssertLessThan(result, 4)
    }
  }

  func testSampleTokenWithFullTopP() {
    // Test categorical sampling path (topP >= 1)
    let logits = MLXArray([Float(0.1), Float(0.2), Float(0.3), Float(0.4)])
    MLX.eval(logits)

    let config = GenerationConfig(
      temperature: 0.7,
      topP: 1.0, // topP >= 1 triggers categorical sampling
      repetitionPenalty: nil
    )

    for _ in 0..<10 {
      let result = sampleToken(logits: logits, config: config, previousTokens: [])
      XCTAssertGreaterThanOrEqual(result, 0)
      XCTAssertLessThan(result, 4)
    }
  }
}
