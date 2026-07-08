"""
Objectives:By the end of the lesson you should answer:
    What is a function?
    Why use functions?
    How do functions improve AI systems?
    What are parameters?
    What are return values?
    What is scope?
    What are docstrings?

FUNCTIONS IN PYTHON
    A function is a reusable block of code that performs a specific task. 
    It allows you to break down complex problems into smaller, manageable pieces, making your code more organized and easier to read.
    Functions are essential in AI systems because they enable you to modularize your code, making it easier to test and maintain. 
    They also allow you to reuse code, which can save time and reduce errors.
    Parameters are the inputs that a function takes to perform its task. They are defined in the function's signature and can be used within the function to perform operations.
    Return values are the outputs that a function produces after performing its task. They can be used to pass information back to the caller or to other parts of the program.
    Scope refers to the visibility and accessibility of variables within a function. Variables defined within a function are local to that function and cannot be accessed outside of it, while variables defined outside of a function are global and can be accessed from anywhere in the program.
    Docstrings are special strings that are used to document a function. They provide information about what the function does, its parameters, and its return values. Docstrings are typically placed at the beginning of a function and can be accessed using the help() function in Python.    

Benefits:
    1. Reusability
    2. Readability
    3. Modularity
    4. Easier debugging

AI engineers use functions everywhere.
"""

# Basic functions in Python
def greet():
    """
    Displays a greeting message.
    """
    print("Welcome to Python for AI Engineers")


greet()


# Function with parametres
def greet_student(name):
    """
    Greets a specific student.

    Parameter:
        name (str): Student's name
    """
    print(f"Hello {name}")


greet_student("Faisal")

# Multiple parametres
def add_numbers(a, b):
    """
    Adds two numbers.
    """
    result = a + b
    print(result)


add_numbers(10, 20)


# Returen values
def multiply_numbers(a, b):
    """
    Returns multiplication result.
    """
    return a * b


answer = multiply_numbers(5, 4)

print(answer)

# Default parameter values
def welcome(name="Student"):
    """
    Uses default value if no argument supplied.
    """
    print(f"Welcome {name}")


welcome()
welcome("Amina")


# Keyword arguments
def student_info(name, track):
    """
    Display student information.
    """
    print(f"Name: {name}")
    print(f"Track: {track}")


student_info(track="AI/ML", name="John")


# Fuctions calling other functions
def get_name():
    return "Adam Muhammad"


def display_name():
    name = get_name()
    print(name)


display_name()


# Variables and scope
global_variable = "I am global"


def show_scope():
    local_variable = "I am local"

    print(global_variable)
    print(local_variable)


show_scope()


# Docstrings
def square(number):
    """
    Returns the square of a number.

    Args:
        number (int or float)

    Returns:
        int or float
    """
    return number ** 2


print(square(5))

# Example of a function with a docstring that takes a parameter and returns a value
def greeting(name):
    """This function takes a name as a parameter and returns a greeting message."""
    return f"Hello, {name}!"

# Calling the function and printing the result
print(greeting("Alice"))
