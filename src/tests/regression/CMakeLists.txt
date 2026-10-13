# Regression tests: run Meraxes on the Genesis L10_N512 test set (Zenodo,
# L10_N512.zip) for each physics setup in setups/. They need that data, so they
# are only added when MERAXES_TESTDATA_DIR contains the unzipped L10_N512/.
# Run them with `ctest -L regression`; `ctest -LE regression` skips them.
set(MERAXES_TESTDATA_DIR "" CACHE PATH "Directory containing the unzipped L10_N512 regression test data")

if(NOT MERAXES_TESTDATA_DIR)
    message(STATUS "Regression tests disabled (set MERAXES_TESTDATA_DIR to enable them)")
    return()
endif()
if(NOT EXISTS "${MERAXES_TESTDATA_DIR}/L10_N512/a_list.txt")
    message(WARNING "No L10_N512 data in MERAXES_TESTDATA_DIR=${MERAXES_TESTDATA_DIR}; regression tests disabled")
    return()
endif()

# Each setup's reference results come from one build configuration, so a setup
# only runs in that configuration.
set(optional_features USE_MINI_HALOS CALC_MAGS USE_STOCHASTICITY USE_JWST USE_HST)
set(enabled_features "")
foreach(feature IN LISTS optional_features)
    if(${feature})
        list(APPEND enabled_features ${feature})
    endif()
endforeach()

if(enabled_features STREQUAL "")
    set(setups default igm xray xray_metallicity)
elseif(enabled_features STREQUAL "USE_MINI_HALOS")
    set(setups mini_halos)
elseif(enabled_features STREQUAL "CALC_MAGS")
    # calc_mags.par.in has 3 TargetSnaps and 2 rest-frame bands
    if(MAGS_N_SNAPS EQUAL 3 AND MAGS_N_BANDS EQUAL 2)
        set(setups calc_mags)
    else()
        message(STATUS "Regression tests disabled: calc_mags needs MAGS_N_SNAPS=3 and MAGS_N_BANDS=2")
        return()
    endif()
elseif(enabled_features STREQUAL "USE_STOCHASTICITY")
    set(setups stochasticity xray_metallicity_scatter)
else()
    message(STATUS "Regression tests disabled: no setup for the feature combination ${enabled_features}")
    return()
endif()

# The random number sequence on each rank depends on which forests it gets, so
# the references are only valid for this rank count.
set(regression_nproc 2)
set(INPUT_FILE_DIR "${CMAKE_SOURCE_DIR}/input")

# Comparing a run with its reference (compare.py) needs Python with h5py and
# numpy. strict suits CI, whose references come from the same environment;
# loose allows for compiler and library differences on other machines.
set(REGRESSION_COMPARE "loose" CACHE STRING "Regression comparison mode: strict or loose")
set_property(CACHE REGRESSION_COMPARE PROPERTY STRINGS strict loose)
find_package(Python3 COMPONENTS Interpreter)
set(can_compare FALSE)
if(Python3_Interpreter_FOUND)
    execute_process(COMMAND ${Python3_EXECUTABLE} -c "import h5py, numpy"
                    RESULT_VARIABLE python_modules_missing OUTPUT_QUIET ERROR_QUIET)
    if(NOT python_modules_missing)
        set(can_compare TRUE)
    endif()
endif()
if(NOT can_compare)
    message(STATUS "Regression runs enabled, but comparisons disabled: need Python 3 with h5py and numpy")
endif()
# With REGRESSION_PLOTS=ON each comparison also plots the run over its reference
# in plots/ next to summary.h5 (needs matplotlib; CI turns it on and uploads the
# plots). Plotting never changes a test's result. Without it, plot by hand with
# compare.py plot.
option(REGRESSION_PLOTS "Plot each regression run over its reference" OFF)
set(plot_args "")
if(REGRESSION_PLOTS AND can_compare)
    execute_process(COMMAND ${Python3_EXECUTABLE} -c "import matplotlib"
                    RESULT_VARIABLE matplotlib_missing OUTPUT_QUIET ERROR_QUIET)
    if(NOT matplotlib_missing)
        set(plot_args --plot plots)
    else()
        message(WARNING "REGRESSION_PLOTS=ON but matplotlib is missing; no plots will be made")
    endif()
endif()

foreach(setup IN LISTS setups)
    set(run_dir "${CMAKE_CURRENT_BINARY_DIR}/${setup}")
    configure_file(setups/${setup}.par.in "${run_dir}/input.par" @ONLY)

    add_test(NAME regression_${setup}_clean
             COMMAND ${CMAKE_COMMAND} -E rm -rf output
             WORKING_DIRECTORY "${run_dir}")
    add_test(NAME regression_${setup}
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${regression_nproc} ${MPIEXEC_PREFLAGS}
                     $<TARGET_FILE:meraxes> input.par ${MPIEXEC_POSTFLAGS}
             WORKING_DIRECTORY "${run_dir}")

    set_tests_properties(regression_${setup}_clean PROPERTIES
        LABELS regression FIXTURES_SETUP regression_${setup}_clean)
    set_tests_properties(regression_${setup} PROPERTIES
        LABELS regression FIXTURES_REQUIRED regression_${setup}_clean
        FIXTURES_SETUP regression_${setup}_run TIMEOUT 1800)

    if(can_compare)
        # summary.h5 in the run directory is this run's summary: CI uploads it,
        # and it becomes the new reference when results change on purpose.
        set(reference "${CMAKE_CURRENT_SOURCE_DIR}/reference/${setup}.h5")
        if(EXISTS "${reference}")
            add_test(NAME regression_${setup}_compare
                     COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/compare.py check output "${reference}"
                             --mode ${REGRESSION_COMPARE} --save summary.h5 ${plot_args}
                     WORKING_DIRECTORY "${run_dir}")
        else()
            message(STATUS "No reference for regression setup ${setup}: its test only writes ${run_dir}/summary.h5")
            add_test(NAME regression_${setup}_compare
                     COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/compare.py write output summary.h5 ${plot_args}
                     WORKING_DIRECTORY "${run_dir}")
        endif()
        set_tests_properties(regression_${setup}_compare PROPERTIES
            LABELS regression FIXTURES_REQUIRED regression_${setup}_run)
    endif()
endforeach()
