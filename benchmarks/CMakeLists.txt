# Конфигурация benchmark тестов производительности.
# Copyright (C) 2024 ML Labs Project
#
# МЕТОДОЛОГИЯ БЕНЧМАРКОВ:
# =======================
# - Минимум 10 итераций для статистической значимости
# - Прогрев кеша перед измерениями
# - Отключение CPU frequency scaling
# - Измерение CPU cycles, cache misses через perf
#
# МЕТРИКИ:
# - Throughput (ops/sec)
# - Latency (ns/op)
# - Memory bandwidth (GB/s)
# - Cache efficiency (hit rate %)
#
# БАЗОВЫЕ ЛИНИИ:
# Сохраняются в benchmarks/baseline.json для отслеживания регрессий

find_package(benchmark REQUIRED)

# Основной benchmark исполняемый файл
add_executable(ml_labs_benchmarks)

# Источники бенчмарков
target_sources(ml_labs_benchmarks PRIVATE
    # Тензорные операции
    bench_tensor_creation.cpp
    bench_tensor_arithmetic.cpp
    bench_tensor_gemm.cpp          # General Matrix Multiplication
    bench_tensor_convolution.cpp
    
    # Автодифференцирование
    bench_autodiff_forward.cpp
    bench_autodiff_backward.cpp
    bench_gradient_accumulation.cpp
    
    # Оптимизаторы
    bench_optimizer_step.cpp
    bench_adam_update.cpp
    
    # Memory allocators
    bench_memory_pool.cpp
    bench_arena_allocator.cpp
    
    # Параллелизм
    bench_parallel_reduce.cpp
    bench_thread_pool.cpp
)

# Зависимости
target_link_libraries(ml_labs_benchmarks PRIVATE
    ml_labs::core
    benchmark::benchmark
    benchmark::benchmark_main
    Threads::Threads
)

# Оптимизации для точных измерений
target_compile_options(ml_labs_benchmarks PRIVATE
    -O3                    # Максимальная оптимизация
    -march=native          # Использовать все инструкции CPU
    -mtune=native         
    -fno-omit-frame-pointer # Для профилирования
)

# Отключаем LTO для бенчмарков (искажает измерения)
set_target_properties(ml_labs_benchmarks PROPERTIES
    INTERPROCEDURAL_OPTIMIZATION OFF
)

# Макро-бенчмарки для реальных сценариев
add_executable(ml_labs_macro_benchmarks
    macro/bench_training_epoch.cpp
    macro/bench_inference_batch.cpp
    macro/bench_model_loading.cpp
)

target_link_libraries(ml_labs_macro_benchmarks PRIVATE
    ml_labs::core
    benchmark::benchmark
)

# Custom target для запуска всех бенчмарков
add_custom_target(run_benchmarks
    COMMAND ${CMAKE_COMMAND} -E echo "Disabling CPU frequency scaling..."
    COMMAND sudo cpupower frequency-set -g performance || true
    
    COMMAND ${CMAKE_COMMAND} -E echo "Running micro benchmarks..."
    COMMAND $<TARGET_FILE:ml_labs_benchmarks> 
        --benchmark_counters_tabular=true
        --benchmark_out=${CMAKE_BINARY_DIR}/bench_results_micro.json
        --benchmark_out_format=json
        
    COMMAND ${CMAKE_COMMAND} -E echo "Running macro benchmarks..."  
    COMMAND $<TARGET_FILE:ml_labs_macro_benchmarks>
        --benchmark_counters_tabular=true
        --benchmark_out=${CMAKE_BINARY_DIR}/bench_results_macro.json
        --benchmark_out_format=json
        
    COMMAND ${CMAKE_COMMAND} -E echo "Restoring CPU governor..."
    COMMAND sudo cpupower frequency-set -g ondemand || true
    
    DEPENDS ml_labs_benchmarks ml_labs_macro_benchmarks
    COMMENT "Running performance benchmarks..."
)

# Target для профилирования с perf
add_custom_target(profile_benchmarks
    COMMAND perf record -g --call-graph=dwarf 
        $<TARGET_FILE:ml_labs_benchmarks>
        --benchmark_filter=BM_TensorGEMM
    COMMAND perf report
    DEPENDS ml_labs_benchmarks
    COMMENT "Profiling benchmarks with perf..."
)

# Генерация flamegraph
add_custom_target(flamegraph
    COMMAND perf record -F 99 -g --call-graph=dwarf
        $<TARGET_FILE:ml_labs_benchmarks>
    COMMAND perf script | ${CMAKE_SOURCE_DIR}/scripts/flamegraph.pl > flamegraph.svg
    COMMAND ${CMAKE_COMMAND} -E echo "Flamegraph saved to flamegraph.svg"
    DEPENDS ml_labs_benchmarks
    COMMENT "Generating flamegraph..."
)
