include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY git@github.com:google/googletest.git
  GIT_TAG main
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(anderson_test
  anderson_test.cpp
)

target_link_libraries(anderson_test
  PRIVATE
  ${PROJECT_NAME}
  gtest_main
  qutility
)

include(GoogleTest)
gtest_discover_tests(anderson_test)