# CMake конфигурация для лабораторных работ.
# Copyright (C) 2024 ML Labs Project
#
# АРХИТЕКТУРА:
# ============
# Каждая лаба - независимый исполняемый файл со своими зависимостями.
# Лабы могут использовать ml_labs::core но не зависят друг от друга.
#
# СОГЛАШЕНИЯ:
# - Имя директории: lab{номер}_{название}
# - Имя таргета: ml_labs_lab{номер}
# - Исполняемый файл: lab{номер}
#
# АВТООБНАРУЖЕНИЕ:
# Новые лабы автоматически подключаются при добавлении директории.

# Функция для регистрации лабораторной работы.
#
# ПАРАМЕТРЫ:
#   LAB_NAME - имя лабы (например, lab01_linear_regression)
#   LAB_NUMBER - номер лабы для сортировки
#
# СОЗДАЁТ:
#   Target: ml_labs_${LAB_NAME}
#   Executable: ${LAB_NAME}

# function - определение пользовательской функции CMake.
# СИГНАТУРА: function(<name> [<arg1> ...])
# ПАРАМЕТРЫ:
#   name - имя функции (становится новой командой CMake)
#   arg1... - формальные параметры функции
# ОБЛАСТЬ ВИДИМОСТИ: Создаёт новую область для локальных переменных
# СПЕЦИАЛЬНЫЕ ПЕРЕМЕННЫЕ:
#   ARGC - количество переданных аргументов
#   ARGV - список всех аргументов
#   ARGN - список дополнительных неименованных аргументов
#   ARGV0, ARGV1, ... - индивидуальные аргументы
# ВОЗВРАТ: Через set(var value PARENT_SCOPE) или return()
# ВЕРСИЯ: Доступна с CMake 2.6
function(add_lab LAB_NAME LAB_NUMBER)
    # message - вывод сообщения в консоль или лог.
    # СИГНАТУРА: message([<mode>] "message text" ...)
    # РЕЖИМЫ:
    #   (none) - важное сообщение, всегда выводится
    #   STATUS - информационное сообщение с префиксом "-- "
    #   WARNING - предупреждение (жёлтый цвет в терминале, если поддерживается)
    #   AUTHOR_WARNING - предупреждение для авторов CMakeLists.txt
    #   SEND_ERROR - ошибка, продолжает конфигурацию но пропускает генерацию
    #   FATAL_ERROR - критическая ошибка, немедленная остановка с кодом 1
    #   DEPRECATION - сообщение об устаревании (контролируется CMAKE_ERROR_DEPRECATED)
    #   VERBOSE - подробное сообщение (CMake 3.15+, показывается с --log-level=VERBOSE)
    #   DEBUG - отладочное сообщение (CMake 3.15+)
    #   TRACE - трассировочное сообщение (CMake 3.15+)
    # ДЕТАЛИ: Аргументы конкатенируются без пробелов
    # ${LAB_NAME} - подстановка значения переменной LAB_NAME
    message(STATUS "Adding lab: ${LAB_NAME}")
    
    # Проверка существования обязательных файлов
    
    # if - условный оператор для ветвления логики.
    # СИГНАТУРА: if(<condition>) ... elseif(<condition>) ... else() ... endif()
    # ОПЕРАТОРЫ УСЛОВИЙ:
    #   NOT - логическое отрицание
    #   AND - логическое И
    #   OR - логическое ИЛИ
    #   EXISTS path - проверка существования файла или директории
    #   IS_DIRECTORY path - проверка что путь является директорией
    #   IS_SYMLINK path - проверка что путь является символической ссылкой
    #   IS_ABSOLUTE path - проверка что путь абсолютный
    #   DEFINED variable - проверка определения переменной
    #   variable STREQUAL string - точное сравнение строк
    #   variable MATCHES regex - сопоставление с регулярным выражением
    #   variable VERSION_LESS version - сравнение версий
    #   variable IN_LIST list - проверка вхождения в список
    # CMAKE_CURRENT_SOURCE_DIR - автоматическая переменная с путём к текущей исходной директории
    # ДЕТАЛИ: Путь где находится обрабатываемый CMakeLists.txt
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${LAB_NAME}/main.cpp")
        # WARNING - режим предупреждения, продолжает выполнение
        # ЭФФЕКТ: Выводит сообщение но не прерывает конфигурацию
        message(WARNING "Lab ${LAB_NAME} missing main.cpp, skipping")
        
        # return - немедленный выход из функции.
        # СИГНАТУРА: return([PROPAGATE var1 [var2...]])
        # ПАРАМЕТРЫ:
        #   PROPAGATE - список переменных для передачи в родительскую область (CMake 3.25+)
        # СЕМАНТИКА: Прекращает выполнение текущей функции
        # ОБЛАСТЬ ВИДИМОСТИ: Восстанавливает область вызывающего кода
        # ВЕРСИЯ: Базовая с CMake 2.6, PROPAGATE с 3.25
        return()
    # endif - обязательное завершение блока if
    # СЕМАНТИКА: Каждый if() должен иметь парный endif()
    endif()
    
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${LAB_NAME}/CMakeLists.txt")
        message(WARNING "Lab ${LAB_NAME} missing CMakeLists.txt, skipping")
        return()
    endif()
    
    # Добавление поддиректории
    
    # add_subdirectory - добавление поддиректории для обработки CMake.
    # СИГНАТУРА: add_subdirectory(source_dir [binary_dir] 
    #                             [EXCLUDE_FROM_ALL] [SYSTEM])
    # ПАРАМЕТРЫ:
    #   source_dir - путь к поддиректории с CMakeLists.txt (относительный или абсолютный)
    #   binary_dir - путь для генерируемых файлов (по умолчанию source_dir)
    #   EXCLUDE_FROM_ALL - не включать таргеты из поддиректории в цель ALL
    #   SYSTEM - пометить include пути как системные (CMake 3.25+)
    # ЭФФЕКТ:
    #   - Обрабатывает CMakeLists.txt в указанной директории
    #   - Создаёт новую область видимости для переменных
    #   - Наследует все переменные из родительской области
    #   - Изменения переменных не влияют на родительскую область (без PARENT_SCOPE)
    # ДЕТАЛИ: CMAKE_CURRENT_SOURCE_DIR и CMAKE_CURRENT_BINARY_DIR изменяются на новые
    # ВЕРСИЯ: Базовая с CMake 2.6, SYSTEM с 3.25
    add_subdirectory(${LAB_NAME})
    
    # Регистрация в списке лаб для отчётов
    
    # set_property - установка произвольного свойства объекта CMake.
    # СИГНАТУРА: set_property(<GLOBAL|DIRECTORY|TARGET|SOURCE|INSTALL|TEST|CACHE>
    #                         [<target>...]
    #                         [DIRECTORY <dir>...]
    #                         [TARGET_DIRECTORY <target>...]
    #                         APPEND [APPEND_STRING]
    #                         PROPERTY <name> [<value>...])
    # ПАРАМЕТРЫ:
    #   GLOBAL - глобальное свойство проекта
    #   DIRECTORY - свойство директории
    #   TARGET - свойство цели сборки
    #   SOURCE - свойство исходного файла
    #   TEST - свойство теста CTest
    #   CACHE - свойство кэшированной переменной
    #   APPEND - добавить значения к существующему свойству
    #   APPEND_STRING - добавить как строку без разделителя списка
    #   PROPERTY - имя устанавливаемого свойства
    #   value - значения свойства
    # ГЛОБАЛЬНЫЕ СВОЙСТВА: Доступны из любой части проекта
    # ML_LABS_ALL_LABS - пользовательское свойство для хранения списка лаб
    # ВЕРСИЯ: Доступна с CMake 2.6
    set_property(GLOBAL APPEND PROPERTY ML_LABS_ALL_LABS ${LAB_NAME})
# endfunction - завершение определения функции
# СЕМАНТИКА: Обязательное завершение блока function()
# ЭФФЕКТ: Восстанавливает область видимости вызывающего кода
endfunction()

# Автоматическое обнаружение всех лабораторных.
# Ищем директории matching pattern lab[0-9]+_*

# file - операции с файловой системой.
# СИГНАТУРА для GLOB: file(GLOB <variable> [RELATIVE <path>] 
#                          [CONFIGURE_DEPENDS] <globbing-expressions>...)
# ПАРАМЕТРЫ:
#   GLOB - операция поиска файлов по шаблону
#   variable - переменная для сохранения результатов
#   RELATIVE path - вернуть относительные пути от указанной директории
#   CONFIGURE_DEPENDS - перезапускать CMake при изменении файлов (CMake 3.12+)
#   globbing-expressions - шаблоны поиска (wildcards)
# ШАБЛОНЫ:
#   * - любая последовательность символов
#   ? - любой одиночный символ
#   [0-9] - любая цифра от 0 до 9
#   [a-z] - любая буква от a до z
# ДРУГИЕ ОПЕРАЦИИ file():
#   READ - чтение файла
#   WRITE - запись в файл
#   APPEND - добавление в файл
#   DOWNLOAD - загрузка из интернета
#   UPLOAD - загрузка на сервер
#   COPY - копирование файлов
#   INSTALL - установка файлов
#   REMOVE - удаление файлов
#   REMOVE_RECURSE - рекурсивное удаление
#   MAKE_DIRECTORY - создание директорий
#   RELATIVE_PATH - вычисление относительного пути
#   TO_CMAKE_PATH - преобразование в CMake путь
#   TO_NATIVE_PATH - преобразование в нативный путь ОС
#   SIZE - получение размера файла
#   TIMESTAMP - получение временной метки
#   GENERATE - генерация файла с подстановками
#   LOCK - блокировка файла
#   ARCHIVE_CREATE - создание архива
#   ARCHIVE_EXTRACT - распаковка архива
# ПРЕДУПРЕЖДЕНИЕ: GLOB не отслеживает изменения файловой системы автоматически
# ВЕРСИЯ: Базовая с CMake 2.6, расширения добавлялись постепенно
file(GLOB LAB_DIRS 
    RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} 
    ${CMAKE_CURRENT_SOURCE_DIR}/lab[0-9][0-9]_*
)

# Сортировка по номеру для последовательного выполнения

# list - операции со списками (массивами строк).
# СИГНАТУРА: list(<operation> <list> [<args>...])
# ОПЕРАЦИЯ SORT: list(SORT <list> [COMPARE <compare>] [CASE <case>] [ORDER <order>])
# ПАРАМЕТРЫ для SORT:
#   list - список для сортировки (изменяется in-place)
#   COMPARE - метод сравнения:
#     STRING - лексикографическое сравнение (по умолчанию)
#     FILE_BASENAME - сравнение базовых имён файлов
#     NATURAL - естественная сортировка (числа как числа) (CMake 3.18+)
#   CASE - чувствительность к регистру:
#     SENSITIVE - учитывать регистр (по умолчанию)
#     INSENSITIVE - игнорировать регистр
#   ORDER - порядок сортировки:
#     ASCENDING - по возрастанию (по умолчанию)
#     DESCENDING - по убыванию
# ДРУГИЕ ОПЕРАЦИИ list():
#   LENGTH <list> <var> - получить длину списка
#   GET <list> <index> <var> - получить элемент по индексу
#   APPEND <list> <element>... - добавить в конец
#   PREPEND <list> <element>... - добавить в начало (CMake 3.15+)
#   INSERT <list> <index> <element>... - вставить по индексу
#   FIND <list> <value> <var> - найти индекс элемента
#   REMOVE_ITEM <list> <value>... - удалить по значению
#   REMOVE_AT <list> <index>... - удалить по индексу
#   REMOVE_DUPLICATES <list> - удалить дубликаты
#   REVERSE <list> - развернуть список
#   FILTER <list> <INCLUDE|EXCLUDE> REGEX <regex> - фильтрация по regex
#   TRANSFORM <list> <ACTION> - преобразование элементов (CMake 3.12+)
#   POP_BACK <list> [<var>...] - извлечь с конца (CMake 3.15+)
#   POP_FRONT <list> [<var>...] - извлечь с начала (CMake 3.15+)
#   JOIN <list> <glue> <var> - объединить в строку
#   SUBLIST <list> <begin> <length> <var> - получить подсписок
# ВЕРСИЯ: Базовая с CMake 2.6, расширенная сортировка с 3.13
list(SORT LAB_DIRS)

# Обработка каждой найденной лабы

# foreach - цикл по элементам списка или диапазону.
# СИГНАТУРА: foreach(<loop_var> [LISTS <lists>] [IN [LISTS] <lists>] 
#                                [ITEMS <items>] [RANGE <stop>|<start> <stop> [<step>]])
#            ... 
#            endforeach()
# ВАРИАНТЫ ИСПОЛЬЗОВАНИЯ:
#   foreach(var item1 item2...) - итерация по явным элементам
#   foreach(var IN LISTS list1 list2...) - итерация по спискам
#   foreach(var IN ITEMS item1 item2...) - итерация по элементам (CMake 3.1+)
#   foreach(var RANGE stop) - от 0 до stop включительно
#   foreach(var RANGE start stop [step]) - от start до stop с шагом
#   foreach(var IN ZIP_LISTS list1 list2...) - параллельная итерация (CMake 3.17+)
# ПАРАМЕТРЫ:
#   loop_var - переменная цикла, принимает значение текущего элемента
#   LISTS - ключевое слово для указания списков
#   ITEMS - ключевое слово для указания элементов
#   RANGE - ключевое слово для числового диапазона
# УПРАВЛЕНИЕ ЦИКЛОМ:
#   break() - досрочный выход из цикла
#   continue() - переход к следующей итерации
# ВЕРСИЯ: Базовая с CMake 2.6, IN синтаксис с 3.1
foreach(LAB_DIR ${LAB_DIRS})
    # IS_DIRECTORY - проверка что путь является директорией.
    # СИНТАКСИС: if(IS_DIRECTORY path)
    # СЕМАНТИКА: Возвращает TRUE если путь существует и является директорией
    # ДЕТАЛИ: Работает с абсолютными и относительными путями
    #         Для символических ссылок проверяет конечную цель
    if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${LAB_DIR}")
        # Извлечение номера лабы из имени директории
        
        # string - операции со строками.
        # СИГНАТУРА для REGEX MATCH: 
        #   string(REGEX MATCH <match-regex> <out-var> <input>...)
        # ПАРАМЕТРЫ:
        #   REGEX MATCH - операция поиска по регулярному выражению
        #   match-regex - регулярное выражение для поиска
        #   out-var - переменная для сохранения найденного совпадения
        #   input - входные строки для поиска
        # РЕГУЛЯРНОЕ ВЫРАЖЕНИЕ:
        #   lab - литеральный текст
        #   ([0-9]+) - захватывающая группа с одной или более цифрами
        #   _.* - подчёркивание и любые символы после него
        # CMAKE_MATCH_<n> - автоматические переменные с захваченными группами
        #   CMAKE_MATCH_0 - полное совпадение
        #   CMAKE_MATCH_1 - первая захваченная группа
        #   CMAKE_MATCH_2 - вторая захваченная группа и т.д.
        # ДРУГИЕ ОПЕРАЦИИ string():
        #   FIND <string> <substring> <out-var> - поиск подстроки
        #   REPLACE <match> <replace> <out-var> <input> - замена
        #   REGEX REPLACE - замена по regex
        #   REGEX MATCHALL - найти все совпадения
        #   APPEND <string-var> <input> - добавление к строке
        #   PREPEND <string-var> <input> - добавление в начало (CMake 3.10+)
        #   CONCAT <out-var> <input>... - конкатенация
        #   JOIN <glue> <out-var> <input>... - объединение с разделителем
        #   TOLOWER <string> <out-var> - в нижний регистр
        #   TOUPPER <string> <out-var> - в верхний регистр
        #   LENGTH <string> <out-var> - длина строки
        #   SUBSTRING <string> <begin> <length> <out-var> - подстрока
        #   STRIP <string> <out-var> - удаление пробелов
        #   GENEX_STRIP <string> <out-var> - удаление generator expressions
        #   REPEAT <string> <count> <out-var> - повторение строки
        #   COMPARE <op> <string1> <string2> <out-var> - сравнение
        #   ASCII <number>... <out-var> - из ASCII кодов в строку
        #   HEX <string> <out-var> - в шестнадцатеричное представление (CMake 3.18+)
        #   CONFIGURE <string> <out-var> - подстановка переменных @VAR@
        #   MAKE_C_IDENTIFIER <string> <out-var> - сделать валидным C идентификатором
        #   RANDOM [LENGTH <len>] [ALPHABET <alphabet>] <out-var> - случайная строка
        #   TIMESTAMP <out-var> [<format>] [UTC] - временная метка
        #   UUID <out-var> ... - генерация UUID (CMake 3.1+)
        #   JSON <out-var> ... - работа с JSON (CMake 3.19+)
        # _ - условное имя для переменной когда результат не нужен
        # ВЕРСИЯ: Базовая с CMake 2.6, расширения добавлялись постепенно
        string(REGEX MATCH "lab([0-9]+)_.*" _ ${LAB_DIR})
        
        # set - установка значения переменной.
        # СИГНАТУРА: set(<variable> <value>... [PARENT_SCOPE])
        # ПАРАМЕТРЫ:
        #   variable - имя переменной
        #   value - значения (несколько значений создают список)
        #   PARENT_SCOPE - установить в родительской области видимости
        # CMAKE_MATCH_1 - первая захваченная группа из предыдущего REGEX MATCH
        set(LAB_NUMBER ${CMAKE_MATCH_1})
        
        # Вызов пользовательской функции add_lab
        add_lab(${LAB_DIR} ${LAB_NUMBER})
    endif()
# endforeach - завершение цикла foreach
# СЕМАНТИКА: Обязательное завершение блока foreach()
endforeach()

# Custom target для запуска всех лаб

# add_custom_target - создание пользовательской цели сборки.
# СИГНАТУРА: add_custom_target(Name [ALL] [command1 [args1...]]
#                              [COMMAND command2 [args2...] ...]
#                              [DEPENDS depend depend ...]
#                              [BYPRODUCTS files...]
#                              [WORKING_DIRECTORY dir]
#                              [COMMENT comment]
#                              [JOB_POOL job_pool]
#                              [VERBATIM] [USES_TERMINAL]
#                              [COMMAND_EXPAND_LISTS]
#                              [SOURCES src1 [src2...]])
# ПАРАМЕТРЫ:
#   Name - имя создаваемой цели
#   ALL - включить в цель по умолчанию (собирается при make без аргументов)
#   COMMAND - команда для выполнения (может быть несколько)
#   DEPENDS - зависимости (файлы или другие таргеты)
#   BYPRODUCTS - файлы создаваемые командой (для ninja) (CMake 3.2+)
#   WORKING_DIRECTORY - рабочая директория для выполнения команд
#   COMMENT - описание выводимое при сборке
#   JOB_POOL - пул задач для ограничения параллелизма (CMake 3.15+)
#   VERBATIM - точная передача аргументов без интерпретации shell
#   USES_TERMINAL - использовать терминал напрямую (CMake 3.2+)
#   COMMAND_EXPAND_LISTS - раскрывать списки в аргументах команд (CMake 3.8+)
#   SOURCES - исходные файлы для IDE (не компилируются)
# СЕМАНТИКА: Таргет всегда считается устаревшим (out-of-date)
# ВЫПОЛНЕНИЕ: make run_all_labs или cmake --build . --target run_all_labs
# ВЕРСИЯ: Доступна с CMake 2.6
add_custom_target(run_all_labs
    COMMENT "Running all laboratory works..."
)

# Получение списка всех лаб

# get_property - получение значения свойства объекта.
# СИГНАТУРА: get_property(<variable>
#                         <GLOBAL|DIRECTORY|TARGET|SOURCE|INSTALL|TEST|CACHE|VARIABLE>
#                         [<target>]
#                         [DIRECTORY <dir>]
#                         [TARGET_DIRECTORY <target>]
#                         PROPERTY <name>
#                         [SET|DEFINED|BRIEF_DOCS|FULL_DOCS])
# ПАРАМЕТРЫ:
#   variable - переменная для сохранения значения
#   GLOBAL - глобальное свойство проекта
#   DIRECTORY - свойство директории
#   TARGET - свойство цели сборки
#   SOURCE - свойство исходного файла
#   TEST - свойство теста
#   CACHE - свойство кэшированной переменной
#   VARIABLE - свойство обычной переменной
#   PROPERTY name - имя запрашиваемого свойства
#   SET - вернуть TRUE/FALSE если свойство установлено
#   DEFINED - вернуть TRUE/FALSE если свойство определено
#   BRIEF_DOCS - получить краткую документацию свойства
#   FULL_DOCS - получить полную документацию свойства
# ВОЗВРАЩАЕТ: Значение свойства или пустую строку если не установлено
# ВЕРСИЯ: Доступна с CMake 2.6
get_property(ALL_LABS GLOBAL PROPERTY ML_LABS_ALL_LABS)

foreach(LAB ${ALL_LABS})
    # Создание таргета для запуска каждой лабы
    add_custom_target(run_${LAB}
        # COMMAND - команда для выполнения
        # $<TARGET_FILE:target> - генераторное выражение для пути к исполняемому файлу
        # ГЕНЕРАТОРНЫЕ ВЫРАЖЕНИЯ - вычисляются на этапе генерации
        # СИНТАКСИС: $<выражение:параметры>
        # TARGET_FILE - полный путь к главному файлу таргета
        # ДРУГИЕ TARGET ВЫРАЖЕНИЯ:
        #   $<TARGET_NAME:target> - имя таргета
        #   $<TARGET_LINKER_FILE:target> - файл для линковки (.lib/.so)
        #   $<TARGET_SONAME_FILE:target> - файл с SONAME
        #   $<TARGET_PDB_FILE:target> - файл отладочной информации
        #   $<TARGET_BUNDLE_DIR:target> - директория macOS bundle
        #   $<TARGET_BUNDLE_CONTENT_DIR:target> - Contents директория bundle
        #   $<TARGET_PROPERTY:target,prop> - свойство таргета
        #   $<TARGET_OBJECTS:target> - объектные файлы таргета
        #   $<TARGET_POLICY:target,policy> - значение политики для таргета
        #   $<TARGET_EXISTS:target> - 1 если таргет существует, иначе 0
        #   $<TARGET_NAME_IF_EXISTS:target> - имя если существует, иначе пусто
        #   $<TARGET_COMPILE_PDB_FILE:target> - PDB файл компиляции
        #   $<TARGET_COMPILE_PDB_NAME:target> - имя PDB файла
        #   $<TARGET_FILE_DIR:target> - директория исполняемого файла
        #   $<TARGET_FILE_NAME:target> - имя исполняемого файла
        #   $<TARGET_FILE_BASE_NAME:target> - базовое имя без расширения
        # ВЕРСИЯ: Генераторные выражения с CMake 2.8.0, расширялись постепенно
        COMMAND $<TARGET_FILE:ml_labs_${LAB}>
        
        # DEPENDS - зависимости для правильного порядка сборки
        # СЕМАНТИКА: Таргет run_${LAB} будет пересобран если ml_labs_${LAB} изменился
        DEPENDS ml_labs_${LAB}
        
        # WORKING_DIRECTORY - рабочая директория для выполнения команды
        # CMAKE_BINARY_DIR - автоматическая переменная с путём к корню build директории
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/labs/${LAB}
        
        # COMMENT - описание действия, выводимое при сборке
        COMMENT "Running ${LAB}..."
    )
    
    # add_dependencies - установка зависимостей между таргетами.
    # СИГНАТУРА: add_dependencies(<target> [<target-dependency>]...)
    # ПАРАМЕТРЫ:
    #   target - таргет которому добавляются зависимости
    #   target-dependency - таргеты от которых зависит первый
    # СЕМАНТИКА: target не будет собран пока не собраны все dependencies
    # ОТЛИЧИЕ ОТ target_link_libraries: 
    #   - add_dependencies для порядка сборки
    #   - target_link_libraries для линковки библиотек
    # ИСПОЛЬЗОВАНИЕ: Для custom targets, INTERFACE библиотек, порядка генерации
    # ВЕРСИЯ: Доступна с CMake 2.6
    add_dependencies(run_all_labs run_${LAB})
endforeach()

# Генерация сводного отчёта
# Проверка что список лаб не пустой
if(ALL_LABS)
    message(STATUS "")
    message(STATUS "Discovered laboratory works:")
    foreach(LAB ${ALL_LABS})
        # STATUS с отступом для форматирования списка
        message(STATUS "  - ${LAB}")
    endforeach()
    message(STATUS "")
endif()
