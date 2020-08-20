/*
See LICENSE folder for this sample’s licensing information.

Abstract:
A structure that runs a digit recognizer.
*/

import Accelerate

struct TrainingSample {
    
    // Specify `useClientPointer` so that each filter stores its data internally.
    static var filterParameters = BNNSFilterParameters(
        flags: BNNSFlags.useClientPointer.rawValue,
        n_threads: 1,
        alloc_memory: nil,
        free_memory: nil)
    
    static let batchSize = 32
    
    static let convolutionInputImageWidth = 20
    static let convolutionInputImageHeight = 20
    static let convolutionInputImageChannels = 1
    static let convolutionPadding = 1
    
    static let convolutionOutputImageWidth = 20
    static let convolutionOutputImageHeight = 20
    static let convolutionOutputImageChannels = 32
    
    static let convolutionKernelSize = 3 // 3 x 3 kernel.
    
    static let convolutionInputSize = convolutionInputImageWidth * convolutionInputImageHeight * convolutionInputImageChannels
    static let convolutionOutputSize = convolutionOutputImageWidth * convolutionOutputImageHeight * convolutionOutputImageChannels
    static let convolutionWeightSize = convolutionKernelSize * convolutionKernelSize * convolutionInputImageChannels * convolutionOutputImageChannels
    
    static let poolingOutputImageWidth = convolutionOutputImageWidth / 2
    static let poolingOutputImageHeight = convolutionOutputImageWidth / 2
    static let poolingOutputImageChannels = convolutionOutputImageChannels
    static let poolingOutputSize = poolingOutputImageWidth * poolingOutputImageHeight * poolingOutputImageChannels
    
    static let fullyConnectedOutputWidth = 10
    static let fullyConnectedWeightSize = fullyConnectedOutputWidth * poolingOutputSize
    
    // One-hot labels is a ten element array that contains a `1` at the index
    // of the digit. For example, `3` is represented as
    // `[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]`.
    static let oneHotLabelsWidth = fullyConnectedOutputWidth
    
    static var oneHotLabels = BNNSNDArrayDescriptor.allocateUninitialized(
        scalarType: Float.self,
        shape: .vector(oneHotLabelsWidth),
        batchSize: batchSize)
    
    static let inputArray = BNNSNDArrayDescriptor.allocateUninitialized(
        scalarType: Float.self,
        shape: .imageCHW(convolutionInputImageWidth,
                         convolutionInputImageHeight,
                         convolutionInputImageChannels),
        batchSize: batchSize)
    
    static let convolutionWeightsArrayShape = BNNS.Shape.convolutionWeightsOIHW(
        convolutionKernelSize,
        convolutionKernelSize,
        convolutionInputImageChannels,
        convolutionOutputImageChannels)
    
    static let convolutionWeightsArray = BNNSNDArrayDescriptor.allocate(
        randomIn: Float(-0.5)...0.5,
        shape: convolutionWeightsArrayShape,
        batchSize: batchSize)
    
    static let convolutionBiasArray = BNNSNDArrayDescriptor.allocate(
        repeating: Float(0),
        shape: .vector(convolutionOutputImageChannels),
        batchSize: batchSize)
    
    static let featureMaps = convolutionOutputImageChannels
    
    static let batchNormBetaArray = BNNSNDArrayDescriptor.allocate(
        repeating: Float(0),
        shape: .vector(featureMaps),
        batchSize: batchSize)
    
    static let batchNormGammaArray = BNNSNDArrayDescriptor.allocate(
        repeating: Float(1),
        shape: .vector(featureMaps),
        batchSize: batchSize)
    
    static let batchNormMovingVarianceArray = BNNSNDArrayDescriptor.allocate(
        repeating: Float(1),
        shape: .vector(featureMaps),
        batchSize: batchSize)
    
    static let batchNormMovingMeanArray = BNNSNDArrayDescriptor.allocate(
        repeating: Float(1),
        shape: .vector(featureMaps),
        batchSize: batchSize)
    
    static let batchNormOutputShape = BNNS.Shape.imageCHW(
        convolutionOutputImageWidth,
        convolutionOutputImageHeight,
        convolutionOutputImageChannels)
    
    static let batchNormOutputArray = BNNSNDArrayDescriptor.allocateUninitialized(
        scalarType: Float.self,
        shape: batchNormOutputShape,
        batchSize: batchSize)
    
    static let poolingOutputShape = BNNS.Shape.imageCHW(poolingOutputImageWidth,
                                                        poolingOutputImageHeight,
                                                        poolingOutputImageChannels)
    
    static let poolingOutputArray = BNNSNDArrayDescriptor.allocateUninitialized(
        scalarType: Float.self,
        shape: poolingOutputShape,
        batchSize: batchSize)
    
    static let fullyConnectedWeightsArray = BNNSNDArrayDescriptor.allocate(
        randomIn: Float(-0.5)...0.5,
        shape: .matrixRowMajor(poolingOutputSize,
                               fullyConnectedOutputWidth),
        batchSize: batchSize)
    
    static let fullyConnectedOutputArray = BNNSNDArrayDescriptor.allocateUninitialized(
        scalarType: Float.self,
        shape: .vector(fullyConnectedOutputWidth),
        batchSize: batchSize)
    
    // This sample reduces loss to a single value.
    static let lossOutputWidth = 1
    
    static let lossOutputArray = BNNSNDArrayDescriptor.allocateUninitialized(
        scalarType: Float.self,
        shape: .vector(lossOutputWidth))
    
    static let lossInputGradientArray = BNNSNDArrayDescriptor.allocate(
        repeating: Float(0),
        shape: .vector(fullyConnectedOutputWidth),
        batchSize: batchSize)
    
    static let fullyConnectedInputGradientArray = BNNSNDArrayDescriptor.allocate(
        repeating: Float(0),
        shape: .vector(poolingOutputSize),
        batchSize: batchSize)
    
    static let fullyConnectedWeightGradientArray = BNNSNDArrayDescriptor.allocateUninitialized(
        scalarType: Float.self,
        shape: fullyConnectedWeightsArray.shape)
    
    static let fullyConnectedWeightAccumulator1 = BNNSNDArrayDescriptor.allocate(
        repeating: Float(0),
        shape: fullyConnectedWeightsArray.shape)
    static let fullyConnectedWeightAccumulator2 = BNNSNDArrayDescriptor.allocate(
        repeating: Float(0),
        shape: fullyConnectedWeightsArray.shape)
    
    static let convolutionInputGradientArray = BNNSNDArrayDescriptor.allocate(
        repeating: Float(0),
        shape: inputArray.shape,
        batchSize: batchSize)
    
    static let convolutionWeightGradientArray = BNNSNDArrayDescriptor.allocate(
        repeating: Float(0),
        shape: convolutionWeightsArray.shape)
    
    static let convolutionBiasGradientArray = BNNSNDArrayDescriptor.allocate(
        repeating: Float(0),
        shape: convolutionBiasArray.shape)
    
    static let batchNormBetaGradientArray = BNNSNDArrayDescriptor.allocate(
        repeating: Float(0),
        shape: .vector(featureMaps))
    
    // Accumulators used by optimizer.
    
    static let batchNormGammaGradientArray = BNNSNDArrayDescriptor.allocate(
        repeating: Float(0),
        shape: .vector(featureMaps))
    
    static let convolutionWeightAccumulator1 = BNNSNDArrayDescriptor.allocate(
        repeating: Float(0),
        shape: convolutionWeightsArray.shape)
    static let convolutionWeightAccumulator2 = BNNSNDArrayDescriptor.allocate(
        repeating: Float(0),
        shape: convolutionWeightsArray.shape)
    static let convolutionBiasAccumulator1 = BNNSNDArrayDescriptor.allocate(
        repeating: Float(0),
        shape: convolutionBiasArray.shape)
    static let convolutionBiasAccumulator2 = BNNSNDArrayDescriptor.allocate(
        repeating: Float(0),
        shape: convolutionBiasArray.shape)
    
    static let batchNormBetaAccumulator1 = BNNSNDArrayDescriptor.allocate(
        repeating: Float(0),
        shape: batchNormBetaArray.shape)
    static let batchNormBetaAccumulator2 = BNNSNDArrayDescriptor.allocate(
        repeating: Float(0),
        shape: batchNormBetaArray.shape)
    static let batchNormGammaAccumulator1 = BNNSNDArrayDescriptor.allocate(
        repeating: Float(0),
        shape: batchNormGammaArray.shape)
    static let batchNormGammaAccumulator2 = BNNSNDArrayDescriptor.allocate(
        repeating: Float(0),
        shape: batchNormGammaArray.shape)
    
    static let poolingInputGradientArray = BNNSNDArrayDescriptor.allocate(
        repeating: Float(0),
        shape: batchNormOutputArray.shape,
        batchSize: batchSize)
    
    static func testExample() {
        runTrainingExample()
        
        let score = test()
        
        freeArrayDescriptors()
        
        print("Training sample test accuracy: \(score)")
    }
    
    static func runTrainingExample() {
        let maximumIterationCount = 1000
        
        // An array containing the last `recentLossesCount` losses.
        let recentLossesCount = 20
        var recentLosses = [Float]()
        
        // The loss threshold at which to consider the training phase complete.
        let averageRecentLossThreshold = Float(0.125)
        
        for epoch in 0 ..< maximumIterationCount {
            if epoch == 500 {
                adam.learningRate /= 10
            }
            
            generateInputAndLabels()
            forwardPass()
            computeLoss()
            
            guard let loss = lossOutputArray.makeArray(of: Float.self,
                                                       batchSize: 1)?.first else {
                print("Unable to calculate loss.")
                return
            }
            
            if recentLosses.isEmpty {
                recentLosses = [Float](repeating: loss,
                                       count: recentLossesCount)
            }
            
            recentLosses[epoch % recentLossesCount] = loss
            let averageRecentLoss = vDSP.mean(recentLosses)
            
            if epoch % 10 == 0 {
                print("Epoch \(epoch): \(loss) : \(averageRecentLoss)")
            }
            
            if averageRecentLoss < averageRecentLossThreshold {
                print("Recent average loss: \(averageRecentLoss), breaking at epoch \(epoch).")
                break
            }
            
            backwardPass()
        }
    }
    
    // Perform forward pass by calling `apply` on fused, pooling, and fully
    // connected layers.
    static func forwardPass() {
        do {
            try fusedConvBatchNormLayer.apply(batchSize: batchSize,
                                              input: inputArray,
                                              output: batchNormOutputArray,
                                              for: .training)
            
            try poolingLayer.apply(batchSize: batchSize,
                                   input: batchNormOutputArray,
                                   output: poolingOutputArray)
            
            try fullyConnectedLayer.apply(batchSize: batchSize,
                                          input: poolingOutputArray,
                                          output: fullyConnectedOutputArray)
        } catch {
            fatalError("Forward pass failed.")
        }
    }
    
    // Perform backward pass by calling `applyBackward` on fully connected
    // layer, pooling layer, and fused layer. Apply optimizer step to fully
    // connected and fused parameters:
    static func backwardPass() {
        backwardFully()
        optimizerFully()
        backwardPooling()
        backwardFused()
        optimizerFused()
    }
    
    // Compute loss.
    static func computeLoss() {
        do {
            try lossLayer.apply(batchSize: batchSize,
                                input: fullyConnectedOutputArray,
                                labels: oneHotLabels,
                                output: lossOutputArray,
                                generatingInputGradient: lossInputGradientArray)
        } catch {
            fatalError("`loss()` failed.")
        }
    }
    
    // MARK: Backward pass and optimization step
    
    static var adam = BNNS.AdamOptimizer(
        learningRate: 0.01,
        beta1: 0.9,
        beta2: 0.999,
        timeStep: 1,
        epsilon: 1e-07,
        gradientScale: 1,
        regularizationScale: 0.01,
        clipsGradientsTo: -0.5 ... 0.5,
        regularizationFunction: BNNSOptimizerRegularizationL2)
    
    // Backward apply of fully connected layer, generating gradient for
    // weights.
    static func backwardFully() {
        do {
            try fullyConnectedLayer.applyBackward(
                batchSize: batchSize,
                input: poolingOutputArray,
                output: fullyConnectedOutputArray,
                outputGradient: lossInputGradientArray,
                generatingInputGradient: fullyConnectedInputGradientArray,
                generatingWeightsGradient: fullyConnectedWeightGradientArray)
        } catch {
            fatalError("`backwardFully()` failed.")
        }
    }
    
    // Apply an optimizer step to the fully connected weights.
    static func optimizerFully() {
        do {
            try adam.step(parameters: [fullyConnectedWeightsArray],
                          gradients: [fullyConnectedWeightGradientArray],
                          accumulators: [fullyConnectedWeightAccumulator1,
                                         fullyConnectedWeightAccumulator2],
                          filterParameters: filterParameters)
        } catch {
            fatalError("`optimizerFully()` failed.")
        }
    }
    
    // Backward apply of pooling layer.
    static func backwardPooling() {
        do {
            try poolingLayer.applyBackward(
                batchSize: batchSize,
                input: batchNormOutputArray,
                output: poolingOutputArray,
                outputGradient: fullyConnectedInputGradientArray,
                generatingInputGradient: poolingInputGradientArray)
        } catch {
            fatalError("`backwardPooling()` failed.")
        }
    }
    
    // Backward apply of fused layer, generating gradients for convolution
    // weights and bias, and batch normalisation beta and gamma
    static func backwardFused() {
        do {
            let gradientParameters = [convolutionWeightGradientArray,
                                      convolutionBiasGradientArray,
                                      batchNormBetaGradientArray,
                                      batchNormGammaGradientArray]
            
            try fusedConvBatchNormLayer.applyBackward(
                batchSize: batchSize,
                input: inputArray,
                output: batchNormOutputArray,
                outputGradient: poolingInputGradientArray,
                generatingInputGradient: convolutionInputGradientArray,
                generatingParameterGradients: gradientParameters)
        } catch {
            fatalError("`backwardFused()` failed.")
        }
    }
    
    // Apply an optimizer step to the convolution weights and bias, and the
    // batch normalisation beta and gamma.
    static func optimizerFused() {
        do {
            try adam.step(
                parameters: [convolutionWeightsArray, convolutionBiasArray,
                             batchNormBetaArray, batchNormGammaArray],
                gradients: [convolutionWeightGradientArray, convolutionBiasGradientArray,
                            batchNormBetaGradientArray, batchNormGammaGradientArray],
                accumulators: [convolutionWeightAccumulator1, convolutionBiasAccumulator1,
                               batchNormBetaAccumulator1, batchNormGammaAccumulator1,
                               convolutionWeightAccumulator2, convolutionBiasAccumulator2,
                               batchNormBetaAccumulator2, batchNormGammaAccumulator2],
                filterParameters: filterParameters)
        } catch {
            fatalError("`optimizerFused()` failed.")
        }
        adam.timeStep += 1
    }
}

