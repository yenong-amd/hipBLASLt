/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once

#include <array>
#include <vector>
#include <memory>

#include "DataTypes_Half.hpp"

namespace TensileLite
{
    /**
     * \ingroup Tensile
     * \defgroup MLPClassification MLP Classification
     *
     * @brief Classification model using multilayer perceptron
     *
     * Neural net used to estimate efficiency values for solutions in the
     * library. Used for MLPClassificationLibrary.
     */

    /**
     * \ingroup MLPClassification
     */
    namespace MLPClassification
    {

        // using dtype = TensileLite::Half;
        using dtype = float;
        // using dtype = _Float16;   // very slow, -mavx512fp16
        // using dtype = __bf16;     // compiler errors, tried with -march=native

        struct StandardScaler
        {
            void operator()(std::vector<dtype>& F) const;

            std::vector<dtype> mean, scale;
        };

        struct WeightMatrix
        {
            WeightMatrix() = default;
            WeightMatrix(const std::vector<float>& W) : weight(W.begin(), W.end()) {}
            virtual ~WeightMatrix() = default;

            virtual void operator()(const std::vector<dtype>& F,
                                    std::vector<dtype>& Fout) const;

            std::vector<dtype> weight;
        };

        /*
         * Specifying matrix dimensions at compile time for better unrolling etc.?
         */
        template <int N_IN>
        struct WeightMatrixFixed : public WeightMatrix
        {
            WeightMatrixFixed() = default;
            WeightMatrixFixed(const std::vector<float>& W) : WeightMatrix(W) {}

            void operator()(const std::vector<dtype>& F,
                            std::vector<dtype>& Fout) const override;
        };

        struct DenseLayer
        {
            DenseLayer() = default;
            DenseLayer(const std::vector<float>& weights, std::vector<float>& bias);

            std::vector<dtype>
            operator()(const std::vector<dtype>& F) const
            {
                auto Fout = B;
                (*W)(F, Fout);
                return Fout;
            }

            std::vector<dtype> B;
            std::shared_ptr<WeightMatrix> W;
        };

        struct ResBlock
        {
            ResBlock() = default;

            std::vector<dtype>
            operator()(const std::vector<dtype>& F) const;

            DenseLayer linear1, linear2, res;
        };

        struct TunaNet
        {
            TunaNet() = default;

            std::vector<dtype> predict(std::vector<float> const& probkey) const;

            std::string description() const
            {
                return "TunaNet";
            }

            std::vector<ResBlock> res_blocks;
            DenseLayer dense;
            StandardScaler scaler;
        };

     } // namespace MLPClassification
} // namespace TensileLite
