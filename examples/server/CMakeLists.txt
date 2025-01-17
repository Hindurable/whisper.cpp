set(TARGET whisper-server)


# Find Boost
find_package(Boost REQUIRED COMPONENTS system thread)

add_executable(${TARGET}
    server.cpp
    websocket_handler.cpp
    httplib.h)

include(DefaultTargetOptions)

target_link_libraries(${TARGET} PRIVATE 
    common 
    json_cpp 
    whisper 
    ${CMAKE_THREAD_LIBS_INIT}
    Boost::system
    Boost::thread
    z
    ssl
    crypto)

if (WIN32)
    target_link_libraries(${TARGET} PRIVATE ws2_32)
endif()

install(TARGETS ${TARGET} RUNTIME)
