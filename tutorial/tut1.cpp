#include <iostream>
#include "app_config.hpp"




/*
 * Turorial 1: runtime configuration
 *
 * This shows how to use Mara's runtime configuration data structures,
 *
 * - mara::config_template_t
 * - mara::config_string_map_t
 * - mara::config_t
 * 
 */




//=============================================================================
int main(int argc, const char* argv[])
{
    // Step 1: create an empty mara::config_template_t using the
    // mara::make_config_template function:
    auto cfg_template1 = mara::make_config_template();


    // Step 2: populate the template with the names and default values of your
    // runtime parameters:
    cfg_template1 = cfg_template1.item("resolution", 1024);
    cfg_template1 = cfg_template1.item("tfinal", 10.0);
    cfg_template1 = cfg_template1.item("outdir", "my_data");


    // This config template is now configured to load parameters named
    // 'resolution', 'tfinal', and 'outdir', with types int, double, and
    // std::string respectively. Note that the template instance is immutable:
    // it returns a new copy each time you add an item. Also note that the
    // parameter types are inferred from from the value, so if you want a
    // parameter with type double, then make sure the default value has a
    // decimal point in it. You would typically combine the above steps like
    // this:
    auto cfg_template = mara::make_config_template()
    .item("resolution", 1024)
    .item("tfinal", 10.0)
    .item("outdir", "my_data");


    // Step 3: create a mara::config_t instance from the template, containing
    // all the default parameter values:
    auto run_config1 = cfg_template.create();


    // You can now update the run_config with values you get from the command
    // line, from a parameter file, or from an HDF5 checkpoint file. For
    // example, to update your config with values from the command line, you can
    // create a mara::config_string_map_t like this (this takes any items in the
    // argv array that have the form key=val and returns them as a string ->
    // string map):
    auto args = mara::argv_to_string_map(argc, argv);


    // Step 4: now you can update the run config with that map:
    run_config1 = run_config1.update(args);


    // If an argument was given whose key is not registered in the template,
    // you'll get an exception. Note that the mara::config_t is also immutable,
    // so you have to re-assign it to a new variable (or itself) each time you
    // update it. You can also set values manually:
    run_config1 = run_config1.set("resolution", 4096);


    // All of the above steps could also be combined into one:
    auto run_config = cfg_template.create().update(args);


    // Once you have loaded runtime parameters from all your parameter sources,
    // you will pass the config to your problem initialization code. Before you
    // do that, you might want to print the run config to the terminal to give
    // the user some feedback on what they're about to run:
    mara::pretty_print(std::cout, "config", run_config);


    // You can try this out on the command line by typing e.g.
    //
    // $> ./tut1 tfinal=5.0 outdir=my_project/data

    return 0;
}
