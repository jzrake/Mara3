/**
 ==============================================================================
 Copyright 2019, Magdalena Siwek

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


#include <cstdio>
#include "core_ndarray.hpp"
#include <cmath>
#include "core_ndarray_ops.hpp"
#include "core_hdf5.hpp"
#include "app_serialize.hpp"
#include <sstream>

struct state_t
{
    int iteration = 0;
    double time = 0.0;
    nd::shared_array<double,1> vertices;
    //nd::shared_array<double,1> cell_centers;
    nd::shared_array<double,1> concentration;
};

auto midpoint = [] (auto array)
{
    auto L = array | nd::select_axis(0).from(0).to(1).from_the_end();
    auto R = array | nd::select_axis(0).from(1).to(0).from_the_end();
    return (L + R) * 0.5;
};

auto difference = [] (auto array)
{
    auto L = array | nd::select_axis(0).from(0).to(1).from_the_end();
    auto R = array | nd::select_axis(0).from(1).to(0).from_the_end();
    return L - R;
};

auto advance_concentration(const state_t& state, double dt)
{
    //std::printf("%d\n", state.concentration.size());
    //a is the wave speed
    double a = 1.0;
    // U is the concentration, extend here by 0s on each side to later fit the array
    auto U = state.concentration | nd::extend_periodic_on_axis(0);
    // Flux below
    auto F = U * a; //F: flux in each cell
    //now obtain intercell fluxes
    auto FL = F | nd::select_axis(0).from(0).to(1).from_the_end(); //flux on left
    auto FR = F | nd::select_axis(0).from(1).to(0).from_the_end(); //flux on right
    //the below means: if ADVECTION_WAVESPEED < 0, then evaluate FR, else FL.
    auto Fhat = a < 0 ? FR : FL; // using the “ternary" operator

    //now find the difference between values in Fhat:
    auto Fhat_diff = difference(Fhat);
    //auto new_concentration = state.concentration + Fhat_diff*dt;

    return state.concentration + Fhat_diff * dt | nd::to_shared();
    //new_concentration;
}

state_t create_my_initial_state()
{
    auto sigma = 0.2;
    std::size_t num_cells = 100;
    auto vertices = nd::linspace(-1,1,num_cells+1);
    auto cell_centers = midpoint(vertices); // has size num_cells
    auto concentration = cell_centers | nd::map([sigma] (double x) { return std::exp(-x * x / sigma / sigma);});
    //auto concentration = vertices | nd::map([sigma] (auto x) { return std::exp(-x * x / sigma / sigma );});
    return{ 0, 0.0, vertices.shared(), concentration.shared()};
}

state_t next(const state_t& state)
{
    double dt = 0.1;
    return{
        state.iteration + 1,
        state.time + dt,
        state.vertices,
        //state.concentration
        advance_concentration(state,dt),
        //in advance_concentration, put in the physics (M_i = M...)
    };
}

int main()
{

    system("exec rm -r examples/advect_1d_out/u*.h5");
    //auto x = nd::linspace(0,1,100);
    auto state = create_my_initial_state();
    while (state.time < 200.)
    {
        std::ostringstream os;
        os << "examples/advect_1d_out/u_" << state.iteration << ".h5";
        std::string s = os.str();
        auto h5f = h5::File(s, "w");
        h5f.write("x", midpoint(state.vertices).shared());
        h5f.write("u", state.concentration.shared());

        state = next(state);
        std::printf("%d %lf \n", state.iteration, state.time);
        //auto xx = nd::linspace(-1, 1, 100);
        //std::string s = "u_%d.h5", %state.iteration;
        //const char* c = s.c_str();
        //std::ostringstream os;
        //os << "examples/advect_1d_out/u_" << state.iteration << ".h5";
        //std::string s = os.str();
        //auto h5f = h5::File(s, "w");
        //auto h5f = h5::File("u.h5", "w");
        //h5f.write("xx", xx.shared());
        //h5f.write("x", midpoint(state.vertices).shared());
        //h5f.write("u", state.concentration.shared());
        //h5f.write("time", state.time.shared());
    }

    return 0;
}
