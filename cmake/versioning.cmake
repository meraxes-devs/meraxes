find_package(Git REQUIRED)

execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_VARIABLE GITREF
    OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${GIT_EXECUTABLE} --no-pager diff --no-color
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_VARIABLE GITDIFF
    OUTPUT_STRIP_TRAILING_WHITESPACE)

if (NOT GITDIFF STREQUAL "")
    # configure_file(... ESCAPE_QUOTES) handles quotes; escape the other C-string
    # characters here so dirty-tree provenance retains literal backslashes.
    string(REPLACE "\\" "\\\\" GITDIFF "${GITDIFF}")
    string(REPLACE "\r" "\\r" GITDIFF "${GITDIFF}")
    string(REPLACE "\n" "\\n" GITDIFF "${GITDIFF}")
    string(REPLACE "\t" "\\t" GITDIFF "${GITDIFF}")
endif()

# Make a plain `make`/`ninja` re-run this configure step (and so pick up a new
# commit) automatically, without needing to invoke cmake by hand: HEAD itself
# only changes on checkout, so track the reflog too, which is touched by
# every commit on the current branch as well.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git/HEAD")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/.git/HEAD")
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git/logs/HEAD")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/.git/logs/HEAD")
endif()
