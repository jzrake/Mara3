/**
 ==============================================================================
 Copyright 2019, Jonathan Zrake

 Permission is hereby granted, free of charge, to any person obtaining a copy of
 this software and associated documentation files (the "Software"), to deal in
 the Software without restriction, including without limitation the rights to
 use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 of the Software, and to permit persons to whom the Software is furnished to do
 so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.

 ==============================================================================
*/




#pragma once
#include <chrono>
#include <utility>




//=============================================================================
namespace mara
{
    struct perf_diagnostics_t;

    template<typename Function, typename... Args>
    auto time_execution(Function&& func, Args&&... args);

    // This function does not really belong here... move it to functools or
    // something when that's written.
    template<typename F, typename G>
    auto compose(F f, G g)
    {
        return [f, g] (auto&&... args)
        {
            return f(g(std::forward<decltype(args)>(args)...));
        };
    };
}




//=============================================================================
struct mara::perf_diagnostics_t
{
    perf_diagnostics_t() {}

    template<typename SolverDuration>
    perf_diagnostics_t(SolverDuration duration)
    {
        execution_time_ms = 1e-6 * std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
    }
    double execution_time_ms = 0;
};




template<typename Function, typename... Args>
auto mara::time_execution(Function&& func, Args&&... args)
{
    auto start = std::chrono::high_resolution_clock::now();
    auto result = std::forward<Function>(func)(std::forward<Args>(args)...);
    auto stop = std::chrono::high_resolution_clock::now();
    return std::make_pair(std::move(result), perf_diagnostics_t(stop - start));
};
