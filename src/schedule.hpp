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
#include <map>
#include <string>




//=============================================================================
namespace mara
{
    class task_schedule_t;
}




//=============================================================================
class mara::task_schedule_t
{
public:

    //=========================================================================
    struct task_t
    {
        std::string name;
        int num_times_performed = 0;
        double last_performed = 0.0;
    };

    void create(std::string name, double last_performed=0.0)
    {
        tasks[name] = {name, 0, last_performed};
    }

    int num_times_performed(std::string task_name) const
    {
        throw_unless_task_exists(task_name);
        return tasks.at(task_name).num_times_performed;
    }

    double last_performed(std::string task_name) const
    {
        throw_unless_task_exists(task_name);
        return tasks.at(task_name).last_performed;
    }

    auto increment(std::string task_name, double interval=0.0) const
    {
        auto result = *this;
        result.increment(task_name, interval);
        return result;
    }

    void increment(std::string task_name, double interval=0.0)
    {
        throw_unless_task_exists(task_name);
        tasks.at(task_name).num_times_performed += 1;
        tasks.at(task_name).last_performed += interval;
    }

    auto size() const { return tasks.size(); }
    auto begin() const { return tasks.begin(); }
    auto end() const { return tasks.end(); }
    auto count(std::string task_name) const { return tasks.count(task_name); }

    const auto& at(std::string task_name) const
    {
        throw_unless_task_exists(task_name);
        return tasks.at(task_name);
    }

private:
    //=========================================================================
    void throw_unless_task_exists(std::string task_name) const
    {
        if (! count(task_name))
        {
            throw std::out_of_range("no task scheduled with the name " + task_name);
        }
    }
    std::map<std::string, task_t> tasks;
};