find_package(Criterion)

if(CRITERION_FOUND)
    # add_executable(test_init test_init.c)
    # set_property(TARGET test_init PROPERTY C_STANDARD 99)
    # target_include_directories(test_init PRIVATE ${CRITERION_INCLUDE_DIR} ${CMAKE_SOURCE_DIR})
    # target_link_libraries(test_init PRIVATE ${CRITERION_LIBRARY} meraxes_lib)
    # add_test(NAME test_init COMMAND test_init)

    add_executable(test_parse_snaplist test_parse_snaplist.c)
    set_property(TARGET test_parse_snaplist PROPERTY C_STANDARD 99)
    target_include_directories(test_parse_snaplist PRIVATE ${CRITERION_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR})
    target_link_libraries(test_parse_snaplist PRIVATE ${CRITERION_LIBRARY} meraxes_lib)
    add_test(NAME test_parse_snaplist COMMAND test_parse_snaplist)

    add_executable(test_cooling test_cooling.c)
    set_property(TARGET test_cooling PROPERTY C_STANDARD 99)
    target_include_directories(test_cooling PRIVATE ${CRITERION_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR})
    target_link_libraries(test_cooling PRIVATE ${CRITERION_LIBRARY} meraxes_lib)
    target_compile_definitions(test_cooling PRIVATE COOLING_FUNCS_DIR="${CMAKE_SOURCE_DIR}/input/cooling_functions")
    add_test(NAME test_cooling COMMAND test_cooling)

    add_executable(test_blackhole_feedback test_blackhole_feedback.c)
    set_property(TARGET test_blackhole_feedback PROPERTY C_STANDARD 99)
    target_include_directories(test_blackhole_feedback PRIVATE ${CRITERION_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR})
    target_link_libraries(test_blackhole_feedback PRIVATE ${CRITERION_LIBRARY} meraxes_lib)
    add_test(NAME test_blackhole_feedback COMMAND test_blackhole_feedback)

    add_executable(test_emission_lines test_emission_lines.c)
    set_property(TARGET test_emission_lines PROPERTY C_STANDARD 99)
    target_include_directories(test_emission_lines PRIVATE ${CRITERION_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR})
    target_link_libraries(test_emission_lines PRIVATE ${CRITERION_LIBRARY} meraxes_lib)
    add_test(NAME test_emission_lines COMMAND test_emission_lines)
else()
    message(WARNING "Failed to find Criterion. Criterion unit tests will not be built.")
endif(CRITERION_FOUND)

# Standalone test driver for cooling module (no Criterion required)
add_executable(main_cooling main_cooling.c)
set_property(TARGET main_cooling PROPERTY C_STANDARD 99)
target_include_directories(main_cooling PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR})
target_link_libraries(main_cooling PRIVATE meraxes_lib)
target_compile_definitions(main_cooling PRIVATE COOLING_FUNCS_DIR="${CMAKE_SOURCE_DIR}/input/cooling_functions")

# Standalone test driver for blackhole feedback module (no Criterion required)
add_executable(main_blackhole_feedback main_blackhole_feedback.c)
set_property(TARGET main_blackhole_feedback PROPERTY C_STANDARD 99)
target_include_directories(main_blackhole_feedback PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR})
target_link_libraries(main_blackhole_feedback PRIVATE meraxes_lib)

# Regression tests on the L10_N512 test set (no Criterion needed)
add_subdirectory(regression)

# Tiny MPI integration checks also cover ranks with no galaxies or FFT slabs.
# Compile the independent legacy filter with the production arithmetic options.
# In Release, comparing fast-math production code with a strict-math oracle is not bitwise meaningful.
set_source_files_properties(reference_filter.c PROPERTIES
    COMPILE_OPTIONS "$<TARGET_PROPERTY:meraxes_lib,COMPILE_OPTIONS>")
add_executable(test_cpu_mpi test_cpu_mpi.c reference_filter.c)
set_property(TARGET test_cpu_mpi PROPERTY C_STANDARD 99)
target_include_directories(test_cpu_mpi PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR})
target_link_libraries(test_cpu_mpi PRIVATE meraxes_lib)
foreach(nproc 1 2 3)
    add_test(NAME cpu_mpi_${nproc}
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${nproc} ${MPIEXEC_PREFLAGS}
                     $<TARGET_FILE:test_cpu_mpi> ${MPIEXEC_POSTFLAGS})
    set_tests_properties(cpu_mpi_${nproc} PROPERTIES
        LABELS "cpu;mpi" TIMEOUT 60 PROCESSORS ${nproc}
        ENVIRONMENT "OMPI_MCA_rmaps_base_oversubscribe=1")
endforeach()

# Check bounded galaxy output buffers, HDF5 table compatibility and tree links.
add_executable(test_save_output test_save_output.c)
set_property(TARGET test_save_output PROPERTY C_STANDARD 99)
target_include_directories(test_save_output PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR})
target_link_libraries(test_save_output PRIVATE meraxes_lib)
foreach(nproc 1 2)
    add_test(NAME save_output_mpi_${nproc}
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${nproc} ${MPIEXEC_PREFLAGS}
                     $<TARGET_FILE:test_save_output> ${MPIEXEC_POSTFLAGS})
    set_tests_properties(save_output_mpi_${nproc} PROPERTIES
        LABELS "cpu;mpi;io" TIMEOUT 60 PROCESSORS ${nproc}
        ENVIRONMENT "OMPI_MCA_rmaps_base_oversubscribe=1")
endforeach()

# Synthetic merger trees exercise actual HDF5/MPI reads and cross-batch FOF links.
add_executable(test_tree_reader_mpi test_tree_reader_mpi.c)
set_property(TARGET test_tree_reader_mpi PROPERTY C_STANDARD 99)
target_include_directories(test_tree_reader_mpi PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR})
target_link_libraries(test_tree_reader_mpi PRIVATE meraxes_lib)
foreach(nproc 1 2 3)
    add_test(NAME tree_reader_mpi_${nproc}
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${nproc} ${MPIEXEC_PREFLAGS}
                     $<TARGET_FILE:test_tree_reader_mpi> ${MPIEXEC_POSTFLAGS})
    set_tests_properties(tree_reader_mpi_${nproc} PROPERTIES
        LABELS "cpu;mpi;io" TIMEOUT 60 PROCESSORS ${nproc}
        ENVIRONMENT "OMPI_MCA_rmaps_base_oversubscribe=1")
endforeach()
