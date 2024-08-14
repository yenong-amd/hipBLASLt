/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include <cstdlib>
#include <string>

#include <Tensile/Singleton.hpp>

namespace Tensile
{
    /**
 * @brief Common place for defining flags which enable debug behaviour.
 */
    class Debug : public LazySingleton<Debug>
    {
    public:
        bool printPropertyEvaluation() const;
        bool printPredicateEvaluation() const;
        bool printDeviceSelection() const;
        bool printCodeObjectInfo() const;

        bool printKernelArguments() const;

        bool printDataInit() const;

        // print tensor dims, strides, memory sizes
        bool printTensorInfo() const;

        // if tensors are printed, use hexadecimal output format
        bool printTensorModeHex() const;

        bool printLibraryVersion() const;

        bool printLookupEfficiency() const;

        bool printWinningKernelName() const;

        bool printSolutionSelectionTime() const;

        bool printLibraryLogicIndex() const;

        bool naivePropertySearch() const;

        bool skipKernelLaunch() const;

        bool enableDebugSelection() const;

        bool useExperimentalSelection() const;

        std::string getMetric() const;

        bool getBenchmark() const;

        int getSolutionIndex() const;

        int getGridbasedTopSols() const;

        bool gridBasedKDTree() const;

        bool gridBasedBatchExp() const;

    private:
        friend LazySingleton<Debug>;

        int         m_value;
        int         m_value2;
        bool        m_naivePropertySearch = false;
        bool        m_debugSelection      = false;
        bool        m_experimentSelection = false;
        int         m_solution_index      = -1;
        std::string m_metric              = "";
        int         m_gridbasedTopSols    = 1;
        bool        m_benchmark           = false;
        bool        m_gridbasedKdTree     = false;
        bool        m_gridbasedBatchExp   = false;

        Debug();
    };
} // namespace Tensile
