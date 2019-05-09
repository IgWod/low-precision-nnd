set(HEADERS "DecoderMain.h" "NeuralDecoder.h" "DenseLayer.h" "NeuralUtilities.h" "Utilities.h" "FixedPoint.h")
set(SOURCE "Utilities.cpp" "NeuralUtilities.cpp" "DenseLayer.cpp" "NeuralDecoder.cpp" "DecoderMain.cpp" "FixedPoint.cpp")

add_executable(Decoder ${SOURCE})

target_link_libraries(Decoder /usr/lib64/libpapi.a)

set("${CMAKE_CXX_FLAGS_RELEASE} -O3 -Wall -xCORE-AVX2")

add_custom_command(TARGET Decoder POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_directory
                   ${CMAKE_SOURCE_DIR}/data
                   ${CMAKE_CURRENT_BINARY_DIR}/data)
