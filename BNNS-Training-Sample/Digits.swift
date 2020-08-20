/*
See LICENSE folder for this sample’s licensing information.

Abstract:
Representations of the digits 0...9 as 6 x 6 matrices.
*/

import Foundation

extension TrainingSample {
    struct Digits {
        static let numberOfDigits = 10
        static let numbers = [zero, one, two, three, four, five, six, seven, eight, nine]
        
        static let zero: [Float] = [0, 0, 1, 1, 0, 0,
                                    0, 1, 0, 0, 1, 0,
                                    0, 1, 0, 0, 1, 0,
                                    0, 1, 0, 0, 1, 0,
                                    0, 1, 0, 0, 1, 0,
                                    0, 0, 1, 1, 0, 0]
        
        static let one: [Float] = [0, 0, 0, 1, 0, 0,
                                   0, 0, 1, 1, 0, 0,
                                   0, 1, 0, 1, 0, 0,
                                   0, 0, 0, 1, 0, 0,
                                   0, 0, 0, 1, 0, 0,
                                   0, 0, 0, 1, 0, 0]
        
        static let two: [Float] = [0, 0, 0, 1, 1, 0,
                                   0, 0, 1, 0, 1, 0,
                                   0, 1, 0, 0, 1, 0,
                                   0, 0, 0, 1, 0, 0,
                                   0, 0, 1, 0, 0, 0,
                                   0, 1, 1, 1, 1, 0]
        
        static let three: [Float] = [0, 1, 1, 1, 1, 0,  // ⚪️⚫️⚫️⚫️⚫️⚪️
                                     0, 0, 0, 0, 1, 0,  // ⚪️⚪️⚪️⚪️⚫️⚪️
                                     0, 1, 1, 1, 1, 0,  // ⚪️⚫️⚫️⚫️⚫️⚪️
                                     0, 0, 0, 0, 1, 0,  // ⚪️⚪️⚪️⚪️⚫️⚪️
                                     0, 1, 1, 1, 1, 0,  // ⚪️⚫️⚫️⚫️⚫️⚪️
                                     0, 0, 0, 0, 0, 0]  // ⚪️⚪️⚪️⚪️⚪️⚪️
        
        static let four: [Float] = [0, 0, 1, 1, 0, 0,
                                    0, 1, 0, 1, 0, 0,
                                    1, 0, 0, 1, 0, 0,
                                    1, 1, 1, 1, 1, 1,
                                    0, 0, 0, 1, 0, 0,
                                    0, 0, 0, 1, 0, 0]
        
        static let five: [Float] = [0, 1, 1, 1, 1, 0,
                                    0, 1, 0, 0, 0, 0,
                                    0, 1, 1, 1, 1, 0,
                                    0, 0, 0, 0, 1, 0,
                                    0, 1, 1, 1, 1, 0,
                                    0, 0, 0, 0, 0, 0]
        
        static let six: [Float] = [0, 1, 1, 1, 1, 0,
                                   0, 1, 0, 0, 0, 0,
                                   0, 1, 0, 0, 0, 0,
                                   0, 1, 1, 1, 1, 0,
                                   0, 1, 0, 0, 1, 0,
                                   0, 1, 1, 1, 1, 0]
        
        static let seven: [Float] = [0, 1, 1, 1, 0, 0,
                                     0, 0, 0, 1, 0, 0,
                                     0, 0, 0, 1, 0, 0,
                                     0, 0, 1, 1, 1, 0,
                                     0, 0, 0, 1, 0, 0,
                                     0, 0, 0, 1, 0, 0]
        
        static let eight: [Float] = [0, 1, 1, 1, 1, 0,
                                     0, 1, 0, 0, 1, 0,
                                     0, 1, 0, 0, 1, 0,
                                     0, 1, 1, 1, 1, 0,
                                     0, 1, 0, 0, 1, 0,
                                     0, 1, 1, 1, 1, 0]
        
        static let nine: [Float] = [0, 0, 1, 1, 1, 0,
                                    0, 0, 1, 0, 1, 0,
                                    0, 0, 1, 1, 1, 0,
                                    0, 0, 0, 0, 1, 0,
                                    0, 0, 0, 0, 1, 0,
                                    0, 0, 0, 0, 1, 0]
    }
}
