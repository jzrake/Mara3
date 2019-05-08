#include <iostream>
#include "ndmpi.hpp"
#include "ndh5.hpp"
#include "ndarray.hpp"
#include "ndarray_ops.hpp"
#include "config.hpp"
#include "serialize.hpp"
#include "schedule.hpp"
#include "performance.hpp"
#include "subprogram.hpp"
#include "parallel.hpp"




static auto config_template()
{
    return mara::make_config_template()
    .item("N", 256)
    .item("procs", 256);
}




//=============================================================================
class subprog_partdom : public mara::sub_program_t
{
public:

    int main(int argc, const char* argv[]) override
    {
        auto args = mara::argv_to_string_map(argc, argv);
        auto opts = config_template().create().update(args);
        auto blocks_shape = mara::propose_block_decomposition<3>(opts.get<int>("procs"));
        auto domain_shape = nd::make_uniform_shape<3>(opts.get<int>("N"));
        auto blocks = mara::create_access_pattern_array(domain_shape, blocks_shape);

        for (auto index : blocks.indexes())
        {
            std::cout << mara::to_string(index) << " ... " << mara::to_string(blocks(index)) << std::endl;
        }



        // auto space = h5::Dataspace::simple(domain_shape);
        // auto file = h5::File("test.h5", "w");
        // file.require_dataset("data", h5::Datatype::native_double(), space);



        return 0;
    }

    std::string name() const override
    {
        return "partdom";
    }
};

std::unique_ptr<mara::sub_program_t> make_subprog_partdom()
{
    return std::make_unique<subprog_partdom>();
}
