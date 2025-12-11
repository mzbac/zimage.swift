import XCTest
import MLX
import Hub
import Logging
@testable import ZImage

/// Integration tests for ZImagePipeline using real model inference.
/// These tests require downloading the 8-bit quantized model (~7.5GB).
/// Run with: xcodebuild test -scheme zimage.swift -destination 'platform=macOS' -only-testing:ZImageIntegrationTests/PipelineIntegrationTests -parallel-testing-enabled NO
final class PipelineIntegrationTests: XCTestCase {

  /// Shared pipeline instance to avoid reloading model for each test
  private static var sharedPipeline: ZImagePipeline?

  /// Project root directory (derived from test file location)
  private static let projectRoot: URL = {
    URL(fileURLWithPath: #file)
      .deletingLastPathComponent()  // Remove PipelineIntegrationTests.swift
      .deletingLastPathComponent()  // Remove ZImageIntegrationTests
      .deletingLastPathComponent()  // Remove Tests -> project root
  }()

  /// Output directory for test-generated images (inside project)
  private static let outputDir: URL = {
    let url = projectRoot
      .appendingPathComponent("Tests")
      .appendingPathComponent("ZImageIntegrationTests")
      .appendingPathComponent("Resources")
    try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
  }()

  /// Initialize shared pipeline once for all tests
  override class func setUp() {
    super.setUp()
    // Skip pipeline creation in CI
    if ProcessInfo.processInfo.environment["CI"] == nil {
      sharedPipeline = ZImagePipeline()
    }
  }

  override class func tearDown() {
    // Clean up shared pipeline
    sharedPipeline = nil
    // Clean up Resources directory after all tests
    try? FileManager.default.removeItem(at: outputDir)
    super.tearDown()
  }

  /// Get the shared pipeline or skip test if not available
  private func getPipeline() throws -> ZImagePipeline {
    guard let pipeline = Self.sharedPipeline else {
      throw XCTSkip("Pipeline not available (likely CI environment)")
    }
    return pipeline
  }

  // MARK: - Basic Generation Tests

  func testBasicGeneration() async throws {
    try skipIfNoGPU()
    let pipeline = try getPipeline()

    let tempOutput = Self.outputDir.appendingPathComponent("test_basic.png")

    let request = ZImageGenerationRequest(
      prompt: "a red apple on a white background",
      width: 512,
      height: 512,
      steps: 9,
      outputPath: tempOutput,
      model: "mzbac/z-image-turbo-8bit"
    )

    let outputURL = try await pipeline.generate(request)

    // Verify output exists
    XCTAssertTrue(FileManager.default.fileExists(atPath: outputURL.path))

    // Verify it's a valid image
    let imageData = try Data(contentsOf: outputURL)
    XCTAssertGreaterThan(imageData.count, 1000) // Should be a reasonable image size
  }

  func testDeterministicSeed() async throws {
    try skipIfNoGPU()
    let pipeline = try getPipeline()

    let tempOutput1 = Self.outputDir.appendingPathComponent("test_seed1.png")
    let tempOutput2 = Self.outputDir.appendingPathComponent("test_seed2.png")

    let seed: UInt64 = 42

    // Generate first image
    let request1 = ZImageGenerationRequest(
      prompt: "a blue cube",
      width: 256,
      height: 256,
      steps: 9,
      seed: seed,
      outputPath: tempOutput1,
      model: "mzbac/z-image-turbo-8bit"
    )
    _ = try await pipeline.generate(request1)

    // Generate second image with same seed
    let request2 = ZImageGenerationRequest(
      prompt: "a blue cube",
      width: 256,
      height: 256,
      steps: 9,
      seed: seed,
      outputPath: tempOutput2,
      model: "mzbac/z-image-turbo-8bit"
    )
    _ = try await pipeline.generate(request2)

    // Both images should exist
    XCTAssertTrue(FileManager.default.fileExists(atPath: tempOutput1.path))
    XCTAssertTrue(FileManager.default.fileExists(atPath: tempOutput2.path))

    // With same seed, images should be identical
    let data1 = try Data(contentsOf: tempOutput1)
    let data2 = try Data(contentsOf: tempOutput2)
    XCTAssertEqual(data1, data2, "Same seed should produce identical images")
  }

  func testVariableDimensions() async throws {
    try skipIfNoGPU()
    let pipeline = try getPipeline()

    let dimensions: [(Int, Int)] = [(512, 512), (768, 768), (512, 768)]

    for (width, height) in dimensions {
      let tempOutput = Self.outputDir.appendingPathComponent("test_dim_\(width)x\(height).png")

      let request = ZImageGenerationRequest(
        prompt: "abstract art",
        width: width,
        height: height,
        steps: 9,
        outputPath: tempOutput,
        model: "mzbac/z-image-turbo-8bit"
      )

      let outputURL = try await pipeline.generate(request)
      XCTAssertTrue(FileManager.default.fileExists(atPath: outputURL.path), "Failed for \(width)x\(height)")
    }
  }

  func testLongPrompt() async throws {
    try skipIfNoGPU()
    let pipeline = try getPipeline()

    let tempOutput = Self.outputDir.appendingPathComponent("test_long.png")

    // Create a long detailed prompt
    let longPrompt = """
    A highly detailed digital painting of a majestic castle perched on a cliff overlooking a vast ocean,
    with dramatic storm clouds gathering in the sky, lightning striking in the distance, waves crashing
    against the rocky shore below, medieval architecture with tall spires and flying buttresses,
    surrounded by lush green forests and winding paths, birds flying in formation, a full moon
    partially visible through the clouds, atmospheric perspective creating depth, cinematic lighting,
    4k resolution, trending on artstation, masterpiece quality
    """

    let request = ZImageGenerationRequest(
      prompt: longPrompt,
      width: 512,
      height: 512,
      steps: 9,
      outputPath: tempOutput,
      model: "mzbac/z-image-turbo-8bit",
      maxSequenceLength: 256
    )

    let outputURL = try await pipeline.generate(request)
    XCTAssertTrue(FileManager.default.fileExists(atPath: outputURL.path))
  }

  // MARK: - Output Validation

  func testOutputFileFormat() async throws {
    try skipIfNoGPU()
    let pipeline = try getPipeline()

    let tempOutput = Self.outputDir.appendingPathComponent("test_format.png")

    let request = ZImageGenerationRequest(
      prompt: "test image",
      width: 256,
      height: 256,
      steps: 9,
      outputPath: tempOutput,
      model: "mzbac/z-image-turbo-8bit"
    )

    let outputURL = try await pipeline.generate(request)

    // Verify PNG signature
    let data = try Data(contentsOf: outputURL)
    let pngSignature: [UInt8] = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]
    let fileSignature = [UInt8](data.prefix(8))
    XCTAssertEqual(fileSignature, pngSignature, "Output should be a valid PNG file")
  }

  // MARK: - Prompt Enhancement Tests

  func testPromptEnhancement() async throws {
    try skipIfNoGPU()

    // Load text encoder and tokenizer for direct enhancement testing
    let snapshot = try await loadSnapshot()
    let modelConfigs = try ZImageModelConfigs.load(from: snapshot)
    let logger = Logger(label: "test.prompt-enhancement")
    let weightsMapper = ZImageWeightsMapper(snapshot: snapshot, logger: logger)
    let quantManifest = weightsMapper.loadQuantizationManifest()

    let tokenizer = try QwenTokenizer.load(from: snapshot.appending(path: "tokenizer"))
    let textEncoder = QwenTextEncoder(
      configuration: .init(
        vocabSize: modelConfigs.textEncoder.vocabSize,
        hiddenSize: modelConfigs.textEncoder.hiddenSize,
        numHiddenLayers: modelConfigs.textEncoder.numHiddenLayers,
        numAttentionHeads: modelConfigs.textEncoder.numAttentionHeads,
        numKeyValueHeads: modelConfigs.textEncoder.numKeyValueHeads,
        intermediateSize: modelConfigs.textEncoder.intermediateSize,
        ropeTheta: modelConfigs.textEncoder.ropeTheta,
        maxPositionEmbeddings: modelConfigs.textEncoder.maxPositionEmbeddings,
        rmsNormEps: modelConfigs.textEncoder.rmsNormEps,
        headDim: modelConfigs.textEncoder.headDim
      )
    )
    let textEncoderWeights = try weightsMapper.loadTextEncoder()
    ZImageWeightsMapping.applyTextEncoder(weights: textEncoderWeights, to: textEncoder, manifest: quantManifest, logger: logger)

    let originalPrompt = "a cat"
    let config = PromptEnhanceConfig(
      maxNewTokens: 512,
      temperature: 0.7,
      topP: 0.9,
      repetitionPenalty: 1.05
    )

    let enhancedPrompt = try textEncoder.enhancePrompt(originalPrompt, tokenizer: tokenizer, config: config)

    // Verify enhancement produced a non-empty result
    XCTAssertFalse(enhancedPrompt.isEmpty, "Enhanced prompt should not be empty")

    // Verify enhanced prompt is longer and more detailed than original
    XCTAssertGreaterThan(enhancedPrompt.count, originalPrompt.count, "Enhanced prompt should be longer than original")

    // Verify enhanced prompt doesn't contain thinking tags (they should be stripped)
    XCTAssertFalse(enhancedPrompt.contains("<think>"), "Enhanced prompt should not contain <think> tag")
    XCTAssertFalse(enhancedPrompt.contains("</think>"), "Enhanced prompt should not contain </think> tag")
  }

  /// Helper to load model snapshot for testing
  private func loadSnapshot() async throws -> URL {
    let hubApi = HubApi.shared
    let repo = Hub.Repo(id: "mzbac/z-image-turbo-8bit")
    let filePatterns = ["*.json", "*.safetensors", "tokenizer/*"]
    return try await hubApi.snapshot(from: repo, matching: filePatterns)
  }

  // MARK: - Helper Functions

  private func skipIfNoGPU() throws {
    // Check if running in CI without GPU
    if ProcessInfo.processInfo.environment["CI"] != nil {
      throw XCTSkip("Skipping GPU-intensive test in CI environment")
    }
  }
}
