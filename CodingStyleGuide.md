# DASPy Coding Style Guide

Like most Python projects, we try to adhere to [PEP 8](https://peps.python.org/pep-0008/) (Style Guide for Python Code) and [PEP 257](https://peps.python.org/pep-0257/) (Docstring Conventions) with the modifications documented here. Be sure to read all documents if you intend to contribute code to DASPy.

## Naming
### Names to Avoid
* single character names except for counters or iterators
* dashes (-) in any package/module name
* **__double_leading_and_trailing_underscore__** names (reserved by Python)
### Naming Convention

* Use meaningful variable/function/method names; these will help other people a lot when reading your code.
* Prepending a single underscore (_) means an object is “internal” / “private”, which means that it is not supposed to be used by end-users and the API might change internally without notice to users (in contrast to API changes in public objects which get handled with deprecation warnings for one release cycle).
* Prepending a double underscore (__) to an instance variable or method effectively serves to make the variable or method private to its class (using name mangling).
* Place related classes and top-level functions together in a module. Unlike Java, there is no need to limit yourself to one class per module.
* Use CamelCase for class names, but snake_case for module names, variables and functions/methods.
