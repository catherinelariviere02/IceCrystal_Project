"""
Crystal2Shape: Information Polyhedron Computation & Simulation Pipeline
-----------------------------------------------------------------------
Author: Sumitava Kundu
Description: Reads CIF/Space Group data, extracts Wyckoff positions, 
computes information polyhedra, and executes HPMC simulations via HOOMD-blue.
"""

import crystal_analyzer

if __name__ == "__main__":
    analyzer = crystal_analyzer.CrystalAnalyzer()
    
    try:
        analyzer.prepare_system() # prepare the system by reading CIF/Space Group data and extracting Wyckoff positions
        analyzer.analyze_symmetry() # analyze the symmetry of the system and determine the reference particle and rcut
        shapes, types, ref_particles, envelop_vertices_list = analyzer.generate_shapes() # generate the information polyhedra for each particle type
        analyzer._JSON_writer(shapes, ref_particles, types) # write the shapes and types to a JSON file for record-keeping and future reference
        print("\n[INFO] Shape computed successfully.")
             
        if analyzer.data["create_unitcell"] == "True":
            analyzer.run_simulations(shapes, types,
                        analyzer.data["sys_prep_pf"], analyzer.data["target_pf"])

        print("\n[SUCCESS] Pipeline completed successfully.")
    except Exception as e:
        print(f"\n[CRITICAL ERROR] {e}")