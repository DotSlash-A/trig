// #include <stdio.h>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <string.h>

// #include "mathslib.h"
#include <ginac/ginac.h>

using namespace GiNac;
using namespace std;

int main() {
    double res_hyp;
    int gcd_val;

    res_hyp = all_ratios_given_two_sides(3.0, 4.0);
    printf("Hypotenuse from all_ratios_given_two_sides(3.0, 4.0): %.1lf\n", res_hyp);

    gcd_val = gcd(8, 18);
    printf("GCD of 8 and 18: %d\n", gcd_val);

    // Example 1 for simplest_form
    char* fraction1 = simplest_form(8, 18);
    if (fraction1 != NULL) {
        printf("Simplest form of 8/18: %s\n", fraction1);
        free(fraction1);
        fraction1 = NULL;
    }

    // Define perpendicular and base for trigonometric ratio example
    int perp_val = 3;
    int base_val = 4;

    if (perp_val >= 0 && base_val >= 0) {
        double hyp_double_val = find_hyp((double)perp_val, (double)base_val);
        int hyp_int_val = (int)round(hyp_double_val);
        
        printf("For triangle with perpendicular = %d, base = %d, hypotenuse = %d (rounded from %.2f)\n",
               perp_val, base_val, hyp_int_val, hyp_double_val);

        if (hyp_int_val > 0) {
            char* sin_frac = simplest_form(perp_val, hyp_int_val);
            if (sin_frac != NULL) {
                printf("Sin fraction (%d/%d) in simplest form: %s\n", perp_val, hyp_int_val, sin_frac);
                
                char* sin_copy = strdup(sin_frac); // Create a copy for strtok
                if (sin_copy != NULL) {
                    char* token = strtok(sin_copy, "/");
                    int numerator = atoi(token);
                    token = strtok(NULL, "/");
                    int denominator = atoi(token);
                    printf("Unpacked: %d/%d\n", numerator, denominator);
                    free(sin_copy);
                }
                
                free(sin_frac);
                sin_frac = NULL;
            }
        } else {
            printf("Cannot calculate sin fraction: hypotenuse is zero or invalid.\n");
        }
    } else {
        printf("Perpendicular and base values must be non-negative.\n");
    }
    
    printf("________________________\n");
    TrigRatios ratios = calculate_trig_ratios_struct(3, 4, 5);
    
    printf("sin = ");
    print_fraction(ratios.sinv);
    printf("\ncos = ");
    print_fraction(ratios.cosv);
    printf("*****************************\n");
    printf("we are using sscanf, giving a string, then tking the number to calcualte following ratios:\n");
    TrigRatios calculated_ratios=all_ratios_given_one_ratio("sin=3/5");
    printf("Given sin=3/5, calculated ratios:\n");
    printf("sin = ");
    print_fraction(calculated_ratios.sinv);
    printf("\ncos = ");
    print_fraction(calculated_ratios.cosv);
    printf("\ntan = ");
    print_fraction(calculated_ratios.tanv);
    printf("\ncosec = ");
    print_fraction(calculated_ratios.cosecv);
    printf("\nsec = ");
    print_fraction(calculated_ratios.secv);
    printf("\ncot = ");
    print_fraction(calculated_ratios.cotv);
    printf("\n");
    
    printf("____________________________\n");   
    const char* expr2 = "cos 60";
    double result2 = evaluate_expression(expr2);
    printf("Value of %s is: %lf\n", expr2, result2);
    
    printf("**************************\n")
    printf("evaluating expressions\n")
    
    ex angle60_rad = 60 * Pi / 180; // This simplifies to Pi/3
    ex angle30_rad = 30 * Pi / 180; // This simplifies to Pi/6
    ex term1 = sin(angle60_rad) * cos(angle30_rad);

    // sin(30°) * cos(60°)
    ex term2 = sin(angle30_rad) * cos(angle60_rad);

    // The full expression: sin(60°)cos(30°) + sin(30°)cos(60°)
    ex trig_expression = term1 + term2;
    cout << "The expression sin(60°)*cos(30°) + sin(30°)*cos(60°) evaluates to: " << trig_expression << endl;

    printf(anglerad)


    return 0;
}
 

typedef struct{
    int numerator;
    int denominator;
} Fraction;


typedef struct{
    Fraction sinv;
    Fraction cosv;
    Fraction tanv;
    Fraction cosecv;
    Fraction secv;
    Fraction cotv;
}TrigRatios;

Fraction create_fraction(int num, int den){
    Fraction f;
    f.numerator = num;
    f.denominator=den;
    return f;
}

void print_fraction(Fraction f) {
    printf("%d/%d", f.numerator, f.denominator);
}

TrigRatios calculate_trig_ratios_struct(int op, int adj,int hyp){
    TrigRatios ratios;
    ratios.sinv = create_fraction(op, hyp);
    ratios.cosv = create_fraction(adj, hyp);
    ratios.tanv = create_fraction(op, adj);
    ratios.cosecv = create_fraction(hyp, op);
    ratios.secv = create_fraction(hyp, adj);
    ratios.cotv = create_fraction(adj, op);
    
    return ratios;
}

TrigRatios all_ratios_given_one_ratio(const char* ratio_str){
    TrigRatios ratios;
    
    ratios.sinv=create_fraction(0,1);
    ratios.cosv=create_fraction(0,1);
    ratios.tanv=create_fraction(0,1);
    ratios.cosecv=create_fraction(0,1);
    ratios.secv=create_fraction(0,1);
    ratios.cotv=create_fraction(0,1);
    
    //parsign strings
    char ratio_name[10]={0};
    char ratio_value[20]={0};
    
    if (sscanf(ratio_str, "%[^=]=%s",ratio_name, ratio_value) !=2){
        printf("Errors: Invalid forma: 'name=value'\n");
        return ratios;
    }
    
    // Extract numerator and denominator
    char* value_copy = strdup(ratio_value);
    if (value_copy == NULL) {
        printf("Error: Memory allocation failed\n");
        return ratios;
    }
    
    char* token = strtok(value_copy, "/");
    if (token == NULL) {
        printf("Error: Invalid fraction format\n");
        free(value_copy);
        return ratios;
    }
    
    int numerator = atoi(token);
    
    token = strtok(NULL, "/");
    if (token == NULL) {
        printf("Error: Invalid fraction format\n");
        free(value_copy);
        return ratios;
    }
    
    int denominator = atoi(token);
    free(value_copy);
    
    if (denominator == 0) {
        printf("Error: Denominator cannot be zero\n");
        return ratios;
    }
    // Calculate all trigonometric ratios based on the given one
    if (strcmp(ratio_name, "sin") == 0) {
        // Given sin = opposite/hypotenuse
        int op = numerator;
        int hyp = denominator;
        
        // Using Pythagorean identity: sin²θ + cos²θ = 1
        // cos = √(1 - sin²)
        // For fractions: If sin = op/hyp, then cos = adj/hyp where adj = √(hyp² - op²)
        int adj_squared = hyp * hyp - op * op;
        
        // Check if we have a valid right triangle
        if (adj_squared <= 0) {
            printf("Error: The given sin value does not represent a valid right triangle\n");
            return ratios;
        }
        
        // We'll use integer approximation for simplicity
        int adj = (int)round(sqrt(adj_squared));
        
        // Now calculate all ratios using the sides
        ratios = calculate_trig_ratios_struct(op, adj, hyp);
    }
    else if (strcmp(ratio_name, "cos") == 0) {
        // Given cos = adjacent/hypotenuse
        int adj = numerator;
        int hyp = denominator;
        
        // Using Pythagorean identity: sin²θ + cos²θ = 1
        // sin = √(1 - cos²)
        // For fractions: If cos = adj/hyp, then sin = op/hyp where op = √(hyp² - adj²)
        int op_squared = hyp * hyp - adj * adj;
        
        // Check if we have a valid right triangle
        if (op_squared <= 0) {
            printf("Error: The given cos value does not represent a valid right triangle\n");
            return ratios;
        }
        
        // We'll use integer approximation for simplicity
        int op = (int)round(sqrt(op_squared));
        
        // Now calculate all ratios using the sides
        ratios = calculate_trig_ratios_struct(op, adj, hyp);
    }
    else if (strcmp(ratio_name, "tan") == 0) {
        // Given tan = opposite/adjacent
        int op = numerator;
        int adj = denominator;
        
        // Calculate hypotenuse using Pythagorean theorem
        int hyp = (int)round(sqrt(op * op + adj * adj));
        
        // Now calculate all ratios using the sides
        ratios = calculate_trig_ratios_struct(op, adj, hyp);
    }
    else if (strcmp(ratio_name, "cosec") == 0) {
        // Given cosec = hypotenuse/opposite
        int hyp = numerator;
        int op = denominator;
        
        // Using Pythagorean identity
        int adj_squared = hyp * hyp - op * op;
        
        // Check if we have a valid right triangle
        if (adj_squared <= 0) {
            printf("Error: The given cosec value does not represent a valid right triangle\n");
            return ratios;
        }
        
        int adj = (int)round(sqrt(adj_squared));
        
        // Now calculate all ratios using the sides
        ratios = calculate_trig_ratios_struct(op, adj, hyp);
    }
    else if (strcmp(ratio_name, "sec") == 0) {
        // Given sec = hypotenuse/adjacent
        int hyp = numerator;
        int adj = denominator;
        
        // Using Pythagorean identity
        int op_squared = hyp * hyp - adj * adj;
        
        // Check if we have a valid right triangle
        if (op_squared <= 0) {
            printf("Error: The given sec value does not represent a valid right triangle\n");
            return ratios;
        }
        
        int op = (int)round(sqrt(op_squared));
        
        // Now calculate all ratios using the sides
        ratios = calculate_trig_ratios_struct(op, adj, hyp);
    }
    else if (strcmp(ratio_name, "cot") == 0) {
        // Given cot = adjacent/opposite
        int adj = numerator;
        int op = denominator;
        
        // Calculate hypotenuse using Pythagorean theorem
        int hyp = (int)round(sqrt(op * op + adj * adj));
        
        // Now calculate all ratios using the sides
        ratios = calculate_trig_ratios_struct(op, adj, hyp);
    }
    else {
        printf("Error: Unknown trigonometric ratio '%s'\n", ratio_name);
    }
    
    return ratios;
}


// Forward declarations
char* simplest_form(int a, int b);

double find_hyp(double perp, double base) {
    double hyp = sqrt(perp * perp + base * base);
    return hyp;
}

double all_ratios_given_two_sides(double op, double adj) {
    double hyp = find_hyp(op, adj);
    
    // Convert doubles to ints for simplest_form
    int op_int = (int)round(op);
    int adj_int = (int)round(adj);
    int hyp_int = (int)round(hyp);
    
    // Declare variables to store the ratio strings
    char *sin_val, *cos_val, *tan_val, *cosec_val, *sec_val, *cot_val;
    
    sin_val = simplest_form(op_int, hyp_int);
    cos_val = simplest_form(adj_int, hyp_int);
    tan_val = simplest_form(op_int, adj_int);
    cosec_val = simplest_form(hyp_int, op_int);
    sec_val = simplest_form(hyp_int, adj_int);
    cot_val = simplest_form(adj_int, op_int);
    
    // Print the ratios
    printf("sin = %s\n", sin_val);
    printf("cos = %s\n", cos_val);
    printf("tan = %s\n", tan_val);
    printf("cosec = %s\n", cosec_val);
    printf("sec = %s\n", sec_val);
    printf("cot = %s\n", cot_val);
    
    // Free allocated memory
    free(sin_val);
    free(cos_val);
    free(tan_val);
    free(cosec_val);
    free(sec_val);
    free(cot_val);
    
    return hyp; // Return the hypotenuse value
}

double evaluate_expression(const char* expression){
    char trig_func[10];
    int angle;
    
    //parse expression
    if (sscanf(expression, "%3s %d", trig_func, &angle) !=2){
        printf("Error, invalud expression, expected format: 'func angle'\n");
        return NAN;
    }
    double angle_rad = angle * M_PI/180.0;
    // Evaluate the expression based on the trigonometric function and angle
    if (strcmp(trig_func, "sin") == 0) {
        if (angle == 0) return 0.0;
        if (angle == 30) return 0.5;
        if (angle == 45) return sqrt(2.0) / 2.0;
        if (angle == 60) return sqrt(3.0) / 2.0;
        if (angle == 90) return 1.0;
        else {
            printf("Error: Unsupported angle for sine function\n");
            return NAN;
        }
    }
    else if (strcmp(trig_func, "cos") == 0) {
        if (angle == 0) return 1.0;
        if (angle == 30) return sqrt(3.0) / 2.0;
        if (angle == 45) return sqrt(2.0) / 2.0;
        if (angle == 60) return 0.5;
        if (angle == 90) return 0.0;
        else {
            printf("Error: Unsupported angle for cosine function\n");
            return NAN;
        }
    }
    else if (strcmp(trig_func, "tan") == 0) {
        if (angle == 0) return 0.0;
        if (angle == 30) return 1.0 / sqrt(3.0);
        if (angle == 45) return 1.0;
        if (angle == 60) return sqrt(3.0);
        if (angle == 90) {
            printf("Error: Tangent of 90 degrees is undefined\n");
            return NAN;
        }
        else {
            printf("Error: Unsupported angle for tangent function\n");
            return NAN;
        }
    }
    else {
        printf("Error: Unsupported trigonometric function\n");
        return NAN;
    }
    
}

int gcd(int a, int b) {
    // Make sure we're working with absolute values
    a = abs(a);
    b = abs(b);
    
    while (b != 0) {
        int temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

char* simplest_form(int a, int b) {
    if (b == 0) {
        fprintf(stderr, "Error: Denominator cannot be zero in simplest_form.\n");
        return NULL; 
    }
    
    // Handle negative numbers properly
    int sign = 1;
    if (a < 0 && b < 0) {
        a = -a;
        b = -b;
    } else if (b < 0) {
        sign = -1;
        b = -b;
        a = -a;
    } else if (a < 0) {
        sign = -1;
        a = -a;
    }
    
    int common_divisor = gcd(a, b);
    int num = sign * (a / common_divisor);
    int den = b / common_divisor;

    int length = snprintf(NULL, 0, "%d/%d", num, den);
    char* frac_string = (char*)malloc(length + 1);

    if (frac_string == NULL) {
        perror("Failed to allocate memory for fraction string");
        return NULL;
    }

    snprintf(frac_string, length + 1, "%d/%d", num, den);
    return frac_string;
}



// Online C++ compiler to run C++ program online
// #include <iostream>

// struct Fraction{
//     int numerator;
//     int denominator;
// }

// struct TrigRatios{
//      Fraction sinv;
//     Fraction cosv;
//     Fraction tanv;
//     Fraction cosecv;
//     Fraction secv;
//     Fraction cotv;
// }



// int main() {
//     // Write C++ code here
//     std::cout << "Try programiz.pro";

//     return 0;
// }