/*
See LICENSE folder for this sample’s licensing information.

Abstract:
Input and label generation extension.
*/

import Accelerate

extension TrainingSample {
    static let printDigits = false
    
    static func generateInputAndLabels() {
        
        // Clear input and one-hot labels arrays.
        vDSP_vclr(inputArray.data!.assumingMemoryBound(to: Float.self), 1,
                  vDSP_Length(inputArray.shape.batchStride * batchSize))
        vDSP_vclr(oneHotLabels.data!.assumingMemoryBound(to: Float.self), 1,
                  vDSP_Length(oneHotLabels.shape.batchStride * batchSize))
        
        // Create typed buffer pointers to input and one-hot labels data.
        let inputBufferPointer = UnsafeMutableBufferPointer<Float>(
            start: inputArray.data!.bindMemory(to: Float.self,
                                               capacity: inputArray.shape.batchStride * batchSize),
            count: inputArray.shape.batchStride * batchSize)
        
        let labelsBufferPointer = UnsafeMutableBufferPointer<Float>(
            start: oneHotLabels.data!.bindMemory(to: Float.self,
                                                 capacity: oneHotLabels.shape.batchStride * batchSize),
            count: oneHotLabels.shape.batchStride * batchSize)
        
        // For each batch, write a random digit to a random position in the
        // 20 x 20 grid.
        for i in 0 ..< batchSize {
            let inputOffset = i * convolutionInputSize
            let randomColumnOffset = Int.random(in: 0 ..< convolutionInputImageWidth - 6)
            let randomRowOffset = Int.random(in: 0 ..< convolutionInputImageHeight - 6)
            let digit = Int.random(in: 0 ..< Digits.numberOfDigits)
            labelsBufferPointer[i * oneHotLabelsWidth + digit] = 1
            let number = Digits.numbers[digit]
            
            for row in 0 ..< 6 {
                let randomIndex = inputOffset + randomColumnOffset + (row + randomRowOffset) * convolutionInputImageWidth
                
                for j in 0 ..< 6 {
                    inputBufferPointer[randomIndex + j] = number[ 6 * row + j]
                }
            }
            
            if printDigits {
                print("\n\n---- \(digit) ----")
                for column in 0 ..< convolutionInputImageWidth {
                    var str = ""
                    for row in 0 ..< convolutionInputImageHeight {
                        let i = inputOffset + row + column * convolutionInputImageWidth
                        str += inputBufferPointer[i].isZero ? "⚪️" : "⚫️"
                    }
                    print(str)
                }
            }
        }
    }
}
