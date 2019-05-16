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
    class schedule_t;
}




//=============================================================================
class mara::schedule_t
{
public:

    //=========================================================================
    struct task_t
    {
        std::string name;
        int num_times_performed = 0;
        double last_performed = 0.0;
        bool is_due = false;
    };

    auto size() const { return tasks.size(); }
    auto begin() const { return tasks.begin(); }
    auto end() const { return tasks.end(); }
    auto count(std::string task_name) const { return tasks.count(task_name); }

    const auto& at(std::string task_name) const
    {
        try {
            return tasks.at(task_name);
        }
        catch (const std::out_of_range&)
        {
            throw std::out_of_range("no task scheduled with the name " + task_name);
        }
    }

    int num_times_performed(std::string task_name) const
    {
        return at(task_name).num_times_performed;
    }

    double last_performed(std::string task_name) const
    {
        return at(task_name).last_performed;
    }

    bool is_due(std::string task_name) const
    {
        return at(task_name).is_due;
    }

    auto& at(std::string task_name)
    {
        try {
            return tasks.at(task_name);
        }
        catch (const std::out_of_range&)
        {
            throw std::out_of_range("no task scheduled with the name " + task_name);
        }        
    }

    void insert(task_t task)
    {
        tasks[task.name] = task;
    }

    void create(std::string task_name)
    {
        tasks[task_name] = {task_name, 0, 0.0, false};
    }

    void mark_as_due(std::string task_name, double amount_to_increase_last_performed_by=0.0)
    {
        at(task_name).is_due = true;
        at(task_name).last_performed += amount_to_increase_last_performed_by;
    }

    void create_and_mark_as_due(std::string task_name)
    {
        create(task_name);
        mark_as_due(task_name);
    }

    void mark_as_completed(std::string task_name)
    {
        at(task_name).is_due = false;        
        at(task_name).num_times_performed += 1;
    }

private:
    //=========================================================================
    std::map<std::string, task_t> tasks;
};
