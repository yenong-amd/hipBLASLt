/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <gtest/gtest.h>

#include <Tensile/ContractionLibrary.hpp>
#include <Tensile/llvm/YAML.hpp>
#include <TestUtils.hpp>

#include "TestData.hpp"

using namespace TensileLite;

/**
 * LibraryPerformanceTest:
 *
 * This suite contains micro-benchmarks for pieces of the runtime library.  It does not
 * exercise any of the Hip-specific code.
 *
 * There are no performance-based assertions or checks.  The timing results are provided by
 * googletest.
 *
 * Most of these tests depend on a library being loaded from a DAT/YAML file.  The library objects
 * are cached so that the deserialization time is not a part of the actual test (outside of the
 * LoadLibrary test). PopulateCache is an empty test whose purpose is to ensure the cache is
 * populated for the actual tests.
 */
struct LibraryPerformanceTest
    : public ::testing::TestWithParam<std::tuple<AMDGPU, std::string, bool, bool>>
{
    AMDGPU                                               hardware;
    std::string                                          filename;
    bool                                                 hasNavi, solutionRequired;
    int                                                      selectionMethod;
    std::shared_ptr<SolutionLibrary<ContractionProblemGemm>> library;

    static std::map<std::string, std::shared_ptr<SolutionLibrary<ContractionProblemGemm>>> libraryCache;

    void SetUp() override
    {
        std::tie(hardware, filename, hasNavi, solutionRequired, selectionMethod) = GetParam();
        setenv TENSILE_SOLUTION_SELECTION_METHOD                                 = selectionMethod;

        if(hardware.processor == AMDGPU::Processor::gfx1010 && !hasNavi)
            GTEST_SKIP();

        library = loadLibrary();

        if(library == nullptr)
        {
            std::cout << libraryPath().native() << std::endl;
            if(!boost::filesystem::is_regular_file(libraryPath()))
                GTEST_SKIP();
            else
                ASSERT_NE(library, nullptr);
        }
    }

    void TearDown() override
    {
        unset TENSILE_SOLUTION_SELECTION_METHOD;
    }

    boost::filesystem::path libraryPath()
    {
        return TestData::Instance().file(filename);
    }

    std::shared_ptr<SolutionLibrary<ContractionProblemGemm>> loadLibrary(bool cache = true)
    {
        if(!cache)
            return loadLibraryNoCache();

        auto pathStr = libraryPath().native();

        auto iter = libraryCache.find(pathStr);
        if(iter != libraryCache.end())
            return iter->second;

        return libraryCache[pathStr] = loadLibraryNoCache();
    }

    std::shared_ptr<SolutionLibrary<ContractionProblemGemm>> loadLibraryNoCache()
    {
        auto path = libraryPath();

        if(boost::filesystem::is_regular_file(path))
            return LoadLibraryFile<ContractionProblemGemm>(path.native());

        return nullptr;
    }
};

std::map<std::string, std::shared_ptr<SolutionLibrary<ContractionProblemGemm>>>
    LibraryPerformanceTest::libraryCache;

TEST_P(LibraryPerformanceTest, PopulateCache)
{
    // Empty test to ensure cache is populated by the SetUp() function.
    // See comment at top of this file.
}

TEST_P(LibraryPerformanceTest, LoadLibrary)
{
    auto library = loadLibrary(false);
}

TEST_P(LibraryPerformanceTest, CreateProblem)
{
    for(int i = 0; i < 1000000; i++)
        RandomGEMM();
}

TEST_P(LibraryPerformanceTest, FindSolution)
{
    for(int i = 0; i < 10; i++)
    {
        auto problem  = RandomGEMM();
        //auto solution = library->findBestSolution(problem, hardware);
        auto solution = library->findTopSolutions(problem, hardware, 1);

        if(solutionRequired)
            EXPECT_EQ(solution.size(), 1) << i << problem;
        // {
        //     for(auto const& kernel : solution)
        //     {
        // std::cout << kernel->description() << std::endl;
        // ASSERT_NE(kernel, nullptr) << i << problem;
        // }
        // }
    }
}

TEST_P(LibraryPerformanceTest, FindCachedSolution)
{
    for(int i = 0; i < 100; i++)
    {
        auto problem  = RandomGEMM();
        auto solution = library->findTopSolutions(problem, hardware, 1);

        if(solutionRequired)
            ASSERT_NE(solution[0], nullptr) << i << problem;
    }

    auto problem = RandomGEMM();

    for(int i = 0; i < 100; i++)
    {
        auto solution = library->findTopSolutions(problem, hardware, 1);

        if(solutionRequired)
            ASSERT_NE(solution[0], nullptr) << i << problem;
    }
}

TEST_P(LibraryPerformanceTest, Solve)
{
    float                                a, b, c, d;
    ContractionProblemGemm               problem;
    std::shared_ptr<ContractionSolution> solution;

    for(int i = 0; i < 10 && solution == nullptr; i++)
    {
        problem  = RandomGEMM();
        solution = library->findBestSolution(problem, hardware);

        if(solutionRequired)
            EXPECT_NE(solution, nullptr) << problem;
    }

    if(solution)
    {
        ContractionInputs inputs{&a, &b, &c, &d, 1.0, float(problem.beta())};
        for(int i = 0; i < 100000; i++)
        {
            solution->solve(problem, inputs, hardware);
        }
    }
}

TEST_P(LibraryPerformanceTest, SolveWithLog)
{
    float                                a, b, c, d;
    ContractionProblemGemm                   problem;
    std::shared_ptr<ContractionSolution> solution;

    for(int i = 0; i < 10 && solution == nullptr; i++)
    {
        problem  = RandomGEMM();
        solution = library->findBestSolution(problem, hardware);

        if(solutionRequired)
            EXPECT_NE(solution, nullptr) << problem;
    }

    if(solution)
    {
        ContractionInputs inputs{&a, &b, &c, &d, 1.0, float(problem.beta())};
        solution->kernelArgsLog = true;
        for(int i = 0; i < 100000; i++)
        {
            solution->solve(problem, inputs, hardware);
        }
    }
}

TEST_P(LibraryPerformanceTest, FindAndSolve)
{
    for(int i = 0; i < 100000; i++)
    {
        auto                          problem  = RandomGEMM();
        auto                          solution = library->findBestSolution(problem, hardware);
        float                         a, b, c, d;
        ContractionInputs             inputs{&a, &b, &c, &d, 1.0f, 1.0f};

        if(solutionRequired)
            ASSERT_NE(solution, nullptr) << i << problem;

        if(solution != nullptr)
            solution->solve(problem, inputs, hardware);
    }
}

TEST_P(LibraryPerformanceTest, FindAndSolveWithLog)
{
    for(int i = 0; i < 100000; i++)
    {
        auto                          problem  = RandomGEMM();
        auto                          solution = library->findBestSolution(problem, hardware);
        float                         a, b, c, d;
        ContractionInputs             inputs{&a, &b, &c, &d, float(1.0), float(problem.beta())};

        if(solutionRequired)
        {
            ASSERT_NE(solution, nullptr) << i << problem;
        }

        if(solution != nullptr)
        {
            solution->kernelArgsLog = true;
            solution->solve(problem, inputs, hardware);
        }
    }
}

TEST_P(LibraryPerformanceTest, SpecificSizes)
{
    // N	N	256	12	1024	1	256	1024	0	256

    auto problem = ContractionProblemGemm::GEMM_Strides(false,
                                                    false,
                                                    DataType::Float,
                                                    DataType::Float,
                                                    DataType::Float,
                                                    DataType::Float,
                                                    256,
                                                    12,
                                                    1024,
                                                    1,
                                                    256,
                                                    1024,
                                                    1024,
                                                    12,
                                                    256,
                                                    12,
                                                    256,
                                                    12,
                                                    2.0);

    auto solution = library->findBestSolution(problem, hardware);
    std::cout << solution->description();
    if(solutionRequired)
        ASSERT_NE(solution, nullptr);
}

std::vector<LibraryPerformanceTest::ParamType> GetLibraries(std::string const& ext)
{
    std::vector<LibraryPerformanceTest::ParamType> rv;

    std::vector<AMDGPU> gpus{AMDGPU(AMDGPU::Processor::gfx900, 64, "Vega 10"),
                             AMDGPU(AMDGPU::Processor::gfx906, 64, "Vega 20")};

    for(auto const& gpu : gpus)
        rv.push_back(std::make_tuple(gpu, "Kernels." + ext, false, false, 0));

    rv.push_back(std::make_tuple(AMDGPU(AMDGPU::Processor::gfx942, 304, "Aquavanjaram"),
                                 "Mlp_Kernels." + ext,
                                 false,
                                 true,
                                 3));
    return rv;
}

std::vector<LibraryPerformanceTest::ParamType> GetParams()
{
    std::vector<LibraryPerformanceTest::ParamType> rv;

#ifdef TENSILE_YAML
    auto yamlParams = GetLibraries("yaml");
    rv.insert(rv.end(), yamlParams.begin(), yamlParams.end());
#endif

#ifdef TENSILE_MSGPACK
    auto datParams = GetLibraries("dat");
    rv.insert(rv.end(), datParams.begin(), datParams.end());
#endif

    return rv;
}

INSTANTIATE_TEST_SUITE_P(LLVM, LibraryPerformanceTest, ::testing::ValuesIn(GetParams()));
