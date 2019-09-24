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
    // const auto child_num = 1 >> tree_rank;

    // auto session = mpi::Session();
    // auto comm    = mpi::comm_world();
    // printf("Running on %d processes\n", comm.size());


    //Build topology of rank tree
    auto to_zeros = [] (int value) { return mara::arithmetic_sequence_t<int, 4>{0, 0, 0, 0}; };
    auto topology = mara::tree_of<tree_rank>(0).bifurcate_all(to_zeros).bifurcate_all(to_zeros);
    auto topology_indexes = topology.indexes();


    //Get hilbert indexes
    auto hindexes = topology_indexes.map([] (auto i) { return mara::global_hilbert_index(i); });


    //Create lists of index-hindex pairs
    auto I = mara::linked_list_t<mara::tree_index_t<tree_rank>>{topology_indexes.begin(), topology_indexes.end()};
    auto H = mara::linked_list_t<int>{hindexes.begin(), hindexes.end()};
    auto hi_pair = H.pair(I);
    
    
    //Sort the pair-list by hilbert index
    auto hi_sorted = hi_pair.sort([] (auto a, auto b) { return a.first < b.first; });
    

    //Convert to nd::array and divvy into equal sized segments
    auto size = 7;
    auto rank_sequences = nd::make_array_from(hi_sorted) | nd::divvy(size);
    

    //Print out ragged array of each rank's assigned hilbert indexes
    for(int i = 0; i < size; i++)
    {
        std::printf("Rank %d: ", i);
        for(int j = 0; j < rank_sequences(i).size(); j++)
        {
            std::printf("%d ", rank_sequences(i)(j).first);
        }
        std::printf("\n");
    }

    return 0;
}