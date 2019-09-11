#include <iomanip>
#include <iostream>
#include <sstream>
#include <cstdio>
#include <cassert>
#include <cmath>

#include "core_mpi.hpp"
#include "core_tree.hpp"
#include "core_ndarray.hpp"
#include "core_ndarray_ops.hpp"
#include "core_linked_list.hpp"


void mpi_hello_world(mpi::Communicator comm)
{
    int rank    = comm.rank();
    int size    = comm.size();

    mpi::printf_master("%s", "I am Master!\n");
    comm.barrier();

    for(int i=0; i < size; i++)
    {
        if (rank==i)
            printf("Hello World from rank %d out of %d\n", rank, size);
    }
}


void mpi_ping_pong(mpi::Communicator comm, int limit)
{

    int count   = 0;
    int rank    = comm.rank();
    int size    = comm.size();
    int partner = (rank + 1) % 2;
   
    if (size > 2)
        throw std::invalid_argument("mpi_test : mpi_ping_pong : size greater than 2");

    int tag = 0;
    while (count < limit)
    {
        if (rank==count % 2)
        {
            count++;
            comm.send(count, partner, tag);
            printf("%d sent count %d to %d\n", rank, count, partner);
        }
        else
        {
            count = comm.recv<int>(partner, tag);
            printf("%d received count %d from %d\n", rank, count, partner);
        }
    }
}


void mpi_ring(mpi::Communicator comm)
{
    int token;
    int rank    = comm.rank();
    int size    = comm.size();

    if (!mpi::is_master())
    {
        token = comm.recv<int>(rank - 1);
        printf("Process %d received token %d from process %d\n", rank, token, rank - 1);
    }
    else
    {
        token = 113;
    }
    comm.send(token, (rank + 1) % size);

    if (mpi::is_master())
    {
        token = comm.recv<int>(size - 1);
        printf("Process %d received token %d from process %d\n", rank, token, size - 1);
    }

    for(int i=0; i < size; i++)
    {
        comm.barrier();
        if (rank==i)
            printf("CHECK: have toke =  %d\n", token);
    }
}



int main(int argc, char* argv[])
{
    // Tree rank and number of children per node
    const auto tree_rank = 2;
    const auto child_num = 1 >> tree_rank;

    auto session = mpi::Session();
    auto comm    = mpi::comm_world();
    printf("Running on %d processes\n", comm.size());


    // 1. Build trivial tree
    auto to_zeros = [] (int value) { return mara::arithmetic_sequence_t<int, 1>{0, 0, 0, 0}; };
    auto nullTree = mara::tree_of<tree_rank>(0).bifurcate_all(to_zeros).bifurcate_all(to_zeros).bifurcate_all(to_zeros);


    // 2. Build tree of linear hilbert indeces
    // TODO: This will work for a tree with unifrom refinement, but for varying refinement 
    //       will need a routine to get the global_hindex
    auto hilbertTree = nullTree.indexes().map([] (auto i) { return mara::hilbert_index(i); });


    // 3. Assign each block a rank
    //      this function is wrong -> use nd::divvy instead
    auto hindex_to_rank = [] (int mpi_size, int num_blocks)
    {
        auto block_per_proc = std::ceil(num_blocks / mpi_size);
        return [block_per_proc] (auto hindex) { return std::floor(hindex / block_per_proc); };
    }; 
    auto treesize = hilbertTree.size();
    auto rankTree = hilbertTree.map(hindex_to_rank(comm.size(), treesize));
    

    // Make tree of empty linked lists
    auto query_neighbors = [] (auto rankTree)
    {
        return [rankTree] (mara::linked_list_t<int> rank_list, mara::tree_index_t<2> i)
        {
            target_indices = mara::arithmetic_sequence_t<mara::tree_index_t<2>, 1>{i.prev_on(0), i.next_on(0), i.prev_on(1), i.next_on(1)};
            return rank_list.append();
        }
    };
    auto commList = nullTree.map([] (auto i) { return mara::linked_list_t<int>{}; })
                            .pair(nullTree.indexes())
                            .apply(query_neighbors(rankTree));

    return 0;
}