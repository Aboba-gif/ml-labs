"""
Conan 2.x рецепт для ML Labs - высокопроизводительной библиотеки машинного обучения.
Copyright (C) 2024 ML Labs Project

АРХИТЕКТУРА ЗАВИСИМОСТЕЙ:
=========================
Уровень 1 (Core):
  - Eigen: BLAS/LAPACK операции, векторизация
  - fmt/spdlog: логирование и форматирование
  
Уровень 2 (ML):
  - xtensor: многомерные массивы в стиле NumPy
  - autodiff: автоматическое дифференцирование
  - dlib: ML алгоритмы и оптимизация
  
Уровень 3 (Infrastructure):
  - protobuf/cereal: сериализация моделей
  - CLI11: парсинг аргументов командной строки
  - TBB/taskflow: параллельные вычисления

ОПТИМИЗАЦИЯ СБОРКИ:
- Бинарный кеш для ускорения CI/CD
- Параллельная сборка зависимостей
- Profile-guided optimization для Release

ССЫЛКИ:
[1] Conan 2.0 Documentation: https://docs.conan.io/2/
[2] "Dependency Management in C++" - CppCon 2023
"""

from conan import ConanFile
from conan.tools.cmake import CMake, CMakeToolchain, CMakeDeps, cmake_layout
from conan.tools.build import check_min_cppstd, can_run
from conan.tools.scm import Git
from conan.tools.files import copy, collect_libs, save, load
from conan.errors import ConanInvalidConfiguration
import os
import platform


class MLLabsConan(ConanFile):
    # Метаданные пакета
    name = "ml_labs"
    version = "0.1.0"
    license = "MIT"
    author = "ML Labs Team"
    url = "https://github.com/yourusername/ml-labs"
    homepage = url
    description = "High-performance machine learning library in modern C++"
    topics = ("machine-learning", "deep-learning", "autodiff", "cpp23", "eigen")
    
    # Настройки сборки
    settings = "os", "compiler", "build_type", "arch"
    
    # Опции пакета с описаниями
    options = {
        "shared": [True, False],           # Тип библиотеки
        "fPIC": [True, False],             # Position Independent Code
        "with_cuda": [True, False],        # CUDA поддержка для GPU
        "with_mkl": [True, False],         # Intel MKL вместо OpenBLAS
        "with_openmp": [True, False],      # OpenMP параллелизация
        "simd_level": ["none", "sse4", "avx2", "avx512", "neon"],  # SIMD инструкции
        "precision": ["single", "double", "mixed"],  # Точность вычислений
        "enable_profiling": [True, False], # Профилирование кода
        "sanitizers": ["none", "address", "thread", "undefined", "memory"],
    }
    
    # Значения по умолчанию
    default_options = {
        "shared": False,            # Static по умолчанию для лучшей оптимизации
        "fPIC": True,              # Нужно для shared libs на Linux
        "with_cuda": False,        # GPU опционально
        "with_mkl": False,         # OpenBLAS по умолчанию (open source)
        "with_openmp": True,       # Параллелизация включена
        "simd_level": "avx2",      # AVX2 есть на большинстве современных CPU (2013+)
        "precision": "single",     # float32 стандарт для ML
        "enable_profiling": False,
        "sanitizers": "none",
    }
    
    # Генераторы для интеграции с CMake
    generators = "CMakeDeps", "CMakeToolchain"
    
    # Экспортируемые файлы
    exports = "LICENSE", "README.md"
    exports_sources = "CMakeLists.txt", "cmake/*", "include/*", "src/*", "tests/*", "benchmarks/*"
    
    # Короткие пути для Windows (ограничение 260 символов)
    short_paths = True
    
    # Минимальный стандарт C++
    @property
    def _min_cppstd(self):
        return "23"  # C++23 для std::expected, std::mdspan
    
    @property
    def _compilers_minimum_version(self):
        """Минимальные версии компиляторов для C++23"""
        return {
            "gcc": "13",
            "clang": "16", 
            "apple-clang": "15",
            "msvc": "193",  # Visual Studio 2022
            "intel-cc": "2024.0",
        }
    
    def validate(self):
        """Валидация конфигурации перед сборкой"""
        # Проверка C++ стандарта
        if self.settings.compiler.get_safe("cppstd"):
            check_min_cppstd(self, self._min_cppstd)
        
        # Проверка версии компилятора
        compiler = str(self.settings.compiler)
        version = str(self.settings.compiler.version)
        
        if compiler in self._compilers_minimum_version:
            min_version = self._compilers_minimum_version[compiler]
            if version < min_version:
                raise ConanInvalidConfiguration(
                    f"{self.name} requires {compiler} >= {min_version} for C++23 support"
                )
        
        # Проверка несовместимых опций
        if self.options.with_cuda and self.options.with_mkl:
            raise ConanInvalidConfiguration(
                "CUDA and MKL are mutually exclusive (use cuBLAS with CUDA)"
            )
        
        # Thread sanitizer несовместим с address sanitizer
        if self.options.sanitizers in ["thread", "memory"] and self.options.shared:
            raise ConanInvalidConfiguration(
                f"{self.options.sanitizers} sanitizer requires static linking"
            )
        
        # Windows не поддерживает fPIC
        if self.settings.os == "Windows":
            del self.options.fPIC
    
    def config_options(self):
        """Настройка опций в зависимости от платформы"""
        if self.settings.os == "Windows":
            del self.options.fPIC
            
        # NEON только для ARM
        if self.settings.arch not in ["armv7", "armv8", "armv8.3"]:
            if "neon" in self.options.simd_level.values:
                self.options.simd_level = "none"
    
    def requirements(self):
        """Объявление зависимостей"""
        # Математические библиотеки - основа для всех операций
        self.requires("eigen/3.4.0", transitive_headers=True)
        
        # BLAS/LAPACK backend
        if self.options.with_mkl:
            self.requires("intel-mkl/2024.0.0")
        else:
            self.requires("openblas/0.3.25")
        
        # Тензорные операции и broadcasting
        self.requires("xtensor/0.24.7")
        self.requires("xtensor-blas/0.20.0")  # BLAS binding для xtensor
        self.requires("xsimd/11.1.0")         # SIMD абстракция
        
        # Автоматическое дифференцирование - ключевая функция для обучения
        self.requires("autodiff/1.1.0")
        
        # ML алгоритмы и утилиты
        self.requires("dlib/19.24.2")
        
        # Форматирование и логирование
        self.requires("fmt/10.2.1")
        self.requires("spdlog/1.13.0")
        
        # Сериализация для сохранения/загрузки моделей
        self.requires("cereal/1.3.2")         # Header-only, быстрая
        self.requires("protobuf/3.21.12")     # Совместимость с TensorFlow/PyTorch
        self.requires("nlohmann_json/3.11.3") # JSON для конфигов
        
        # CLI и конфигурация
        self.requires("cli11/2.4.1")          # Парсинг аргументов
        self.requires("tomlplusplus/3.4.0")   # TOML конфиги
        
        # Параллелизм и асинхронность  
        self.requires("onetbb/2021.11.0")     # Intel TBB для параллелизма
        self.requires("taskflow/3.6.0")       # Task-based параллелизм
        
        # Сетевое взаимодействие (для distributed training)
        self.requires("grpc/1.54.3")
        self.requires("cppzmq/4.10.0")
        
        # Компрессия для checkpoints
        self.requires("zstd/1.5.5")
        self.requires("lz4/1.9.4")
        
        # Профилирование (опционально)
        if self.options.enable_profiling:
            self.requires("tracy/0.10")       # Real-time профилировщик
        
        # CUDA (опционально)
        if self.options.with_cuda:
            self.requires("cuda-toolkit/12.3")
            self.requires("cudnn/8.9.7")
            self.requires("thrust/2.2.0")
    
    def build_requirements(self):
        """Зависимости только для сборки и тестирования"""
        # Система сборки
        self.tool_requires("cmake/3.28.1")
        self.tool_requires("ninja/1.11.1")
        
        # Тестирование
        self.test_requires("gtest/1.14.0")
        self.test_requires("benchmark/1.8.3")
        self.test_requires("catch2/3.5.1")     # Альтернатива GTest
        
        # Property-based testing
        self.test_requires("rapidcheck/0.0.1")
        
        # Code coverage
        if self.settings.build_type == "Debug":
            self.tool_requires("lcov/2.0")
        
        # Статический анализ
        self.tool_requires("cppcheck/2.13")
        
    def layout(self):
        """Определение layout проекта для CMake"""
        cmake_layout(self, src_folder=".")
        
    def generate(self):
        """Генерация файлов для системы сборки"""
        # CMake toolchain
        tc = CMakeToolchain(self)
        
        # Передача опций в CMake
        tc.variables["ML_LABS_WITH_CUDA"] = self.options.with_cuda
        tc.variables["ML_LABS_WITH_MKL"] = self.options.with_mkl
        tc.variables["ML_LABS_SIMD_LEVEL"] = str(self.options.simd_level).upper()
        tc.variables["ML_LABS_PRECISION"] = str(self.options.precision).upper()
        
        # Оптимизации компилятора для Release
        if self.settings.build_type == "Release":
            if self.settings.compiler == "gcc" or self.settings.compiler == "clang":
                # Агрессивные оптимизации для ML кода
                tc.extra_cxxflags.append("-march=native")
                tc.extra_cxxflags.append("-mtune=native")
                
                # SIMD инструкции
                if self.options.simd_level == "avx512":
                    tc.extra_cxxflags.extend(["-mavx512f", "-mavx512dq", "-mavx512bw", "-mavx512vl"])
                elif self.options.simd_level == "avx2":
                    tc.extra_cxxflags.extend(["-mavx2", "-mfma"])
                elif self.options.simd_level == "sse4":
                    tc.extra_cxxflags.append("-msse4.2")
                
                # Математические оптимизации
                tc.extra_cxxflags.append("-ffast-math")
                tc.extra_cxxflags.append("-fno-math-errno")
                tc.extra_cxxflags.append("-ffinite-math-only")
                
                # Loop оптимизации
                tc.extra_cxxflags.append("-funroll-loops")
                tc.extra_cxxflags.append("-ftree-vectorize")
                
                # Link Time Optimization
                tc.extra_cxxflags.append("-flto=auto")
                tc.extra_ldflags.append("-flto=auto")
                
                # Profile Guided Optimization (если доступно)
                if os.path.exists("profile_data"):
                    tc.extra_cxxflags.append("-fprofile-use=profile_data")
        
        # Debug опции
        elif self.settings.build_type == "Debug":
            tc.extra_cxxflags.extend(["-g3", "-ggdb", "-fno-omit-frame-pointer"])
            
            # Sanitizers
            if self.options.sanitizers == "address":
                tc.extra_cxxflags.append("-fsanitize=address")
                tc.extra_ldflags.append("-fsanitize=address")
            elif self.options.sanitizers == "thread":
                tc.extra_cxxflags.append("-fsanitize=thread")
                tc.extra_ldflags.append("-fsanitize=thread")
            elif self.options.sanitizers == "undefined":
                tc.extra_cxxflags.append("-fsanitize=undefined")
                tc.extra_ldflags.append("-fsanitize=undefined")
            elif self.options.sanitizers == "memory":
                tc.extra_cxxflags.append("-fsanitize=memory")
                tc.extra_ldflags.append("-fsanitize=memory")
        
        tc.generate()
        
        # CMake зависимости
        deps = CMakeDeps(self)
        deps.generate()
    
    def build(self):
        """Сборка проекта"""
        cmake = CMake(self)
        cmake.configure()
        cmake.build()
        
        # Запуск тестов если в Debug режиме
        if self.settings.build_type == "Debug" and can_run(self):
            cmake.test(output_on_failure=True)
    
    def package(self):
        """Создание пакета для распространения"""
        cmake = CMake(self)
        cmake.install()
        
        # Копирование лицензии
        copy(self, "LICENSE", src=self.source_folder, dst=os.path.join(self.package_folder, "licenses"))
        
    def package_info(self):
        """Информация о пакете для потребителей"""
        self.cpp_info.libs = collect_libs(self)
        
        # Определение макросов
        if self.options.with_cuda:
            self.cpp_info.defines.append("ML_LABS_USE_CUDA")
        if self.options.with_mkl:
            self.cpp_info.defines.append("ML_LABS_USE_MKL")
        
        # Флаги компилятора
        if self.settings.os in ["Linux", "FreeBSD"]:
            self.cpp_info.system_libs.extend(["pthread", "m", "dl"])
        
        # CMake targets
        self.cpp_info.set_property("cmake_target_name", "ml_labs::ml_labs")
        self.cpp_info.set_property("cmake_find_mode", "both")
