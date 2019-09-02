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
#include <vector>
#include <mutex>
#include <thread>
#include <future>




//=============================================================================
namespace mara {
    class thread_pool_t;
}




//=============================================================================
class mara::thread_pool_t
{
public:


    struct task_t
    {
        operator bool() const { return run != nullptr; }
        std::function<void(void)> run; 
        std::shared_ptr<std::promise<void>> promise;
    };


    thread_pool_t(int num_workers=4)
    {
        for (int n = 0; n < num_workers; ++n)
        {
            threads.push_back(make_worker(n));
        }
    }


    ~thread_pool_t()
    {
        stop_all();
    }


    void stop_all()
    {
        if (! stop)
        {
            stop = true;
            condition.notify_all();

            for (auto& thread : threads)
            {
                thread.join();
            }
        }
    }


    std::future<void> enqueue(std::function<void(void)> run)
    {
        auto promise = std::make_shared<std::promise<void>>();
        std::lock_guard<std::mutex> lock(mutex);
        pending_tasks.push_back({run, promise});
        condition.notify_one();
        return promise->get_future();
    }


private:


    /**
     * Convenience method to find a task with the given tag.
     */
    std::vector<task_t>::iterator tagged(std::promise<void>* tag, std::vector<task_t>& v)
    {
        return std::find_if(v.begin(), v.end(), [tag] (auto& t) { return t.promise.get() == tag; });
    }


    /**
     * Called by other threads to await the next available task. Pops that
     * task from the queue and returns it. If there was no task, then nullptr
     * is returned. That should only be the case when the pool is shutting
     * down.
     */
    task_t next(int id)
    {
        std::unique_lock<std::mutex> lock(mutex);
        condition.wait(lock, [this] { return stop || ! pending_tasks.empty(); });

        if (pending_tasks.empty())
        {
            return {};
        }

        auto task = pending_tasks.front();
        pending_tasks.erase(pending_tasks.begin());
        running_tasks.push_back(task);

        return task;
    }


    /**
     * Called by other threads to indicate they have finished a task.
     */
    void complete(std::promise<void>* tag)
    {
        std::lock_guard<std::mutex> lock(mutex);
        auto task = tagged(tag, running_tasks);
        task->promise->set_value();
        running_tasks.erase(task);
    }


    /**
     * Called by the constructor to create the workers.
     */
    std::thread make_worker(int id)
    {
        return std::thread([this, id] ()
        {
            while (auto task = next(id))
            {
                task.run();
                complete(task.promise.get());
            }
        });
    }


    std::vector<std::thread> threads;
    std::vector<task_t> pending_tasks;
    std::vector<task_t> running_tasks;
    std::condition_variable condition;
    std::atomic<bool> stop = {false};
    std::mutex mutex;
};