#include <stdio.h>

// Recursive function to print prime factors
void PrimeFactors(int n, int factor) {
    if (n <= 1)
        return;

    if (n % factor == 0) {
        printf("%d ", factor);
        PrimeFactors(n / factor, factor);
    } else {
        PrimeFactors(n, factor + 1);
    }
}

int main() {
    int n;
    printf("Enter a positive integer: ");
    scanf("%d", &n);

    printf("Prime factors of %d are: ", n);
    PrimeFactors(n, 2);

    printf("\n");
    return 0;
}




// #include <stdio.h>


// void primefactors(int n, int factor){
//     if(n<=1){
//         return;
//     }
//     if (n%factor==0){
//         printf("%d ", factor);
//         primefactors(n/factor, factor);
//     } else {
//         primefactors(n, factor+1);
//     }
// }
// int main() {
//     printf("Hello, World!\n");
//     primefactors(100, 2);
//     printf("\n");
//     return 0;
// }
