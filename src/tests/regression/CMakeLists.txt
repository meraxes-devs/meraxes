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
    set(setups default igm xray)
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
    set(setups stochasticity)
else()
    message(STATUS "Regression tests disabled: no setup for the feature combination ${enabled_features}")
    return()
endif()

# The random number sequence on each rank depends on which forests it gets, so
# the references are only valid for this rank count.
set(regression_nproc 2)
set(INPUT_FILE_DIR "${CMAKE_SOURCE_DIR}/input")

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
endforeach()
