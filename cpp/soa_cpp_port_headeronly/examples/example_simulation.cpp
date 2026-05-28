#include <soa/SoaCpp.hpp>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    using namespace soa;

    // Usage:
    //   example_simulation.exe 1
    //   example_simulation.exe 2
    //   example_simulation.exe 5
    int n_links = 1;

    if (argc >= 2) {
        n_links = std::atoi(argv[1]);
    }

    if (n_links < 1) {
        std::cerr << "Number of links must be at least 1.\n";
        return 1;
    }

    std::cout << "Running simulation with " << n_links << " link(s).\n";

    // Link geometry
    Vec3 klOO;
    klOO << 1.0, 0.0, 0.0;

    // Store actual body objects here so pointers remain valid
    std::vector<std::unique_ptr<SOABody>> body_storage;
    body_storage.reserve(n_links);

    // MultibodySystem wants pointers
    std::vector<SOABody*> bodies;
    bodies.reserve(n_links);

    for (int i = 0; i < n_links; ++i) {
        // revy: gravity in z creates torque for a beam initially along x
        Joint joint(klOO, "revy");

        // Aluminum-ish density, rectangular section
        RigidProperties rigid(2700.0, 0.02, 0.02);

        // Flexible beam properties
        // For many links, reduce n_md or dt if it becomes unstable.
        FlexProperties flex(
            9.0e8,   // E
            2.6e7,   // G
            0.1,     // damping
            8,       // nodes per link
            3        // modes per link
        );

        auto body = std::make_unique<SOABody>(joint, rigid, flex);

        Vec tau(1);
        tau << 0.0;
        body->setTau(tau);

        bodies.push_back(body.get());
        body_storage.push_back(std::move(body));
    }

    MultibodySystem system(bodies);
    system.setGravity(true);

    // Time setup
    const double tf = 5;
    const double dt = 0.001;
    const double ani_dt = 0.01;

    Simulation sim(system, tf, dt);
    sim.setAniDt(ani_dt);

    auto start_time = std::chrono::high_resolution_clock::now();

    // Use "BE" if that works best in your current port.
    // If BE becomes slow or unstable, try "RK4".
    sim.integrateSystem("RK4");

    auto end_time = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end_time - start_time;

    std::cout << "Integration took: " << elapsed.count() << " seconds.\n";

    std::string wide_file = "simulation_wide_" + std::to_string(n_links) + "_links.csv";
    std::string nodes_file = "simulation_nodes_" + std::to_string(n_links) + "_links.csv";

    sim.writeCSV(wide_file);
    sim.writeNodeCSV(nodes_file);

    std::cout << "Wrote " << wide_file << "\n";
    std::cout << "Wrote " << nodes_file << "\n";
    std::cout << "Simulation complete.\n";

    return 0;
}