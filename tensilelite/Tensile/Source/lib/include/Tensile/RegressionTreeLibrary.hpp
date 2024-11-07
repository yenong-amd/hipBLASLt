/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <set>
#include <vector>

#include <Tensile/Debug.hpp>
#include <Tensile/MLFeatures.hpp>
#include <Tensile/ProblemKey.hpp>
#include <Tensile/RegressionTree.hpp>
#include <Tensile/SolutionLibrary.hpp>
#include <Tensile/Utils.hpp>

namespace Tensile
{
    /**
     * \ingroup SolutionLibrary
     *
     * Uses a set of regression trees to rank solutions for a given size.
     */

    template <typename MyProblem, typename MySolution = typename MyProblem::Solution>
    struct RegressionTreeLibrary : public SolutionLibrary<MyProblem, MySolution>
    {
        using Forest           = RegressionTree::Forest<std::vector<float>, float>;
        using SolutionFeatures = std::vector<std::shared_ptr<MLFeatures::MLFeature<MySolution>>>;
        using ProblemFeatures  = std::vector<std::shared_ptr<MLFeatures::MLFeature<MyProblem>>>;

        std::map<int, std::shared_ptr<MySolution>> solutions;
        std::shared_ptr<Forest>                    forest;
        SolutionFeatures                           solFeatures;
        ProblemFeatures                            probFeatures;

        static std::string Type()
        {
            return "RegressionTree";
        }
        virtual std::string type() const override
        {
            return Type();
        }
        virtual std::string description() const override
        {
            if(forest == nullptr)
                return concatenate(type(), ", forest: nullptr");
            else
                return concatenate(type(), ": ", forest->description());
        }

        virtual std::shared_ptr<MySolution> getSolutionByIndex(MyProblem const& problem,
                                                               Hardware const&  hardware,
                                                               const int index) const override
        {
            const bool experimental = Debug::Instance().useExperimentalSelection();
            if((!experimental) || (index > solutions.size()) || (index < 0))
            {
                // If the experimental library mode is not on treat it like it asserted out
                return nullptr;
            }

            return nullptr;
        }

        virtual std::shared_ptr<MySolution> findBestSolution(MyProblem const& problem,
                                                             Hardware const&  hardware,
                                                             double*          fitness
                                                             = nullptr) const override
        {
            return nullptr;
        }

        virtual SolutionSet<MySolution>
            findAllSolutions(MyProblem const&          problem,
                             Hardware const&           hardware,
                             SolutionLibrarySearchType searchType
                             = SolutionLibrarySearchType::DEFAULT) const override
        {
            if(searchType != SolutionLibrarySearchType::DEFAULT)
            {
                // if the solution library search is not default then return an empty
                // set of solutions.
                SolutionSet<MySolution> rv;
                return rv;
            }

            const bool experimental = Debug::Instance().useExperimentalSelection();
            if(!experimental)
            {
                // Skip the search for solutions if the environment variable
                // that enables the experimental method is not set
                SolutionSet<MySolution> rv;
                return rv;
            }
            SolutionSet<MySolution> rv;
            for(auto const& row : solutions)
                rv.insert(row.second);

            return rv;
        }

        virtual SolutionVector<MySolution> findTopSolutions(MyProblem const& problem,
                                                            Hardware const&  hardware,
                                                            int numSolutions) const override
        {
            SolutionVector<MySolution> rv;
            return rv;
        }

        virtual SolutionSet<MySolution>
            findAllSolutionsGroupedGemm(std::vector<MyProblem> const& problems,
                                        Hardware const&               hardware,
                                        SolutionLibrarySearchType     searchType
                                        = SolutionLibrarySearchType::DEFAULT) const override
        {
            if(searchType != SolutionLibrarySearchType::DEFAULT)
            {
                // if the solution library search is notSolutionSet default then return an empty
                // set of solutions
                SolutionSet<MySolution> rv;
                return rv;
            }

            const bool experimental = Debug::Instance().useExperimentalSelection();
            if(!experimental)
            {
                // Skip the search for solutions if the environment variable
                // that enables the experimental method is not set
                SolutionSet<MySolution> rv;
                return rv;
            }

            SolutionSet<MySolution> rv;
            for(auto const& row : solutions)
                rv.insert(row.second);

            return rv;
        }
    };

} // namespace Tensile
