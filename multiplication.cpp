#include <iostream>
#include <vector>
#include <complex>
#include <string>
#include <chrono>
#include <omp.h>

/*
 * Исользован материал из https://e-maxx.ru/algo/
 * -Xpreprocessor option
 * Pass option as an option to the preprocessor.
 * You can use this to supply system-specific preprocessor options that GCC does not recognize.
 * clang++ -Xpreprocessor -fopenmp -I/usr/local/include -L/usr/local/lib -lomp --std=c++11 multiplication.cpp -o multiplication
 */

typedef std::complex<double> cd;
typedef std::vector<std::complex<double>> vcd;
typedef std::vector<int> lnum;
const int base = 1e9;

inline
void read_from_str_Base(lnum& a, std::string& s) {
    for (int i = (int)s.length(); i > 0; i -= 9) {
        if (i < 9)
            a.push_back(atoi(s.substr(0, i).c_str()));
        else
            a.push_back(atoi(s.substr(i - 9, 9).c_str()));
    }
    while (a.size() > 1 && a.back() == 0)
        a.pop_back();
}

inline
void read_from_str_FFT(std::vector<int>& a, std::string& s) {
    int n = (int)s.size();
    for (int i = n - 1; i >= 0; --i)
        a.push_back(s[i] - 48);
}

inline
void print_lnum(const lnum& a) {
    printf ("%d", a.empty() ? 0 : a.back());
    for (int i = (int)a.size() - 2; i >= 0; --i)
        printf ("%09d", a[i]);
    printf ("\n");
}

inline
lnum multiply_brute(lnum& a, lnum& b) {
    lnum c(a.size() + b.size());
    for (size_t i = 0; i < a.size(); ++i)
        for (int j = 0, carry = 0; j < (int) b.size() || carry; ++j) {
            long long cur = c[i + j] + a[i] * 1ll * (j < (int) b.size() ? b[j] : 0) + carry;
            c[i + j] = int(cur % base);
            carry = int(cur / base);
        }
    while (c.size() > 1 && c.back() == 0)
        c.pop_back();
    return c;
}

// функция получает вектор коэффициентов и меняет inplace на вектор значений в точках (то же количество)
// или получает вектор значений
void FFT(vcd& p, bool invert) {
    int n = (int) p.size();
    if (n == 1) return; // значение в точке 1 равно свободному члену

    vcd p_0 (n / 2), p_1 (n / 2);
    for (int i = 0; i < n / 2; ++i) {
        p_0[i] = p[2 * i];
        p_1[i] = p[2 * i + 1];
    }

    FFT(p_0, invert);
    FFT(p_1, invert);
    // сейчас у нас есть два вектора значений двух многочленов в корнях 4 степени из единицы
    // в точках 1, i, -1, -i

    double arg = 2 * M_PI / n * (invert ? -1 : 1);
    cd w = 1, wn(cos(arg), sin(arg));
    for (int i = 0; i < n / 2; ++i) {
        p[i] = p_0[i] + w * p_1[i]; // все 4 корня из единицы 4ой степени - это
        p[i + n / 2] = p_0[i] - w * p_1[i]; // квадраты первых 4 корней из единицы 8ой степени
        w *= wn;

        if (invert)
            p[i] /= 2,  p[i + n / 2] /= 2;
    }
}

void fft(std::vector<cd>& a, bool invert) {
    int n = (int) a.size();

    // переставляем элементы в массиве, чтобы работать нерекурсивно inplace
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j >= bit; bit >>= 1)
            j -= bit;
        j += bit;
        if (i < j)
            swap(a[i],a[j]);
    }

    // на примере рассмотрим len = n = 8
    for (int len = 2; len <= n; len <<= 1) {
        double ang = 2 * M_PI / len * (invert ? -1 : 1);
        cd w_len(cos(ang), sin(ang)); // (8_w_{k+1}) = w_len * (8_w_k)
        // 1 итерация тут: y = f(y_even, y_odd)

        // я переставил циклы местами
        cd w(1);
        for (int j = 0; j < len / 2; ++j) {
            for (int i = 0; i < n ; i += len) {
                cd u = a[i + j], v = a[i + j + len / 2] * w;
                a[i + j] = u + v;
                a[i + j + len / 2] = u - v;
            }
            w *= w_len;
        }
    }
    if (invert)
        for (int i = 0; i < n; ++i)
            a[i] /= n;
}

void fft_parallel(std::vector<cd>& a, bool invert) {
    int n = (int) a.size(); // уже степень двойки
    // переставляем элементы в массиве, чтобы работать нерекурсивно inplace
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j >= bit; bit >>= 1)
            j -= bit;
        j += bit;
        if (i < j)
            swap(a[i],a[j]);
    }
    // на примере рассмотрим len = n = 8
    for (int len = 2; len <= n; len <<= 1) {
        double ang = 2 * M_PI / len * (invert ? -1 : 1);
        cd w_len(cos(ang), sin(ang));

        int N = n / len; // количество объединений N, а всего блоков 2 * N

        //omp_set_num_threads(N);
#pragma omp parallel shared(w_len, n, N)    \
                             shared(a, len) num_threads(N > 10 ? 16 : N)
        {
#pragma omp for

            // количество итераций по i есть количество объединений
            // блоков длинны (len / 2) на этапе № lg(len), а всего блоков 2 * n / len
            for (int i = 0; i < n ; i += len) { // хотим распараллелить этот цикл (цикл по объединениям)
                cd w(1);
                // 4 итерации: 4 + 4 корня, где первые 4 корня в квадрате дают
                // все 4е {1, i, -1, -i} корня с предыдущей итерации
                // inplace делаем p(x) = p_even(x^2) + x * p_odd(x^2)
                for (int j = 0; j < len / 2; ++j) { // цикл по корням
                    cd u = a[i + j],  v = a[i + j + len / 2] * w;
                    a[i + j] = u + v;
                    a[i + j + len / 2] = u - v;
                    w *= w_len;
                }
            }

        }
    }
    if (invert)
        for (int i = 0; i < n; ++i)
            a[i] /= n;
}

void multiply_FFT(const std::vector<int> & a, const std::vector<int> & b, std::vector<int> & res) {
    vcd fa (a.begin(), a.end()),  fb (b.begin(), b.end());
    size_t n = 1;
    while (n < std::max(a.size(), b.size()))
        n <<= 1;
    n <<= 1;
    fa.resize (n),  fb.resize (n);

    FFT(fa, false),  FFT(fb, false);
    for (size_t i = 0; i < n; ++i)
        fa[i] *= fb[i];
    FFT(fa, true);

    res.resize (n);
    for (size_t i = 0; i < n; ++i) {
        // это будет работать неправильно для отрицательных коэффициентов в задаче перемножения многочленов
        // res[i] = int(fa[i].real() + 0.5);
        res[i] = (int)lround(fa[i].real());
    }

    int carry = 0;
    for (size_t i = 0; i < n; ++i) {
        res[i] += carry;
        carry = res[i] / 10;
        res[i] %= 10;
    }
}

void multiply_fft(const std::vector<int> & a, const std::vector<int> & b, std::vector<int> & res) {
    vcd fa (a.begin(), a.end()),  fb (b.begin(), b.end());
    size_t n = 1;
    while (n < std::max(a.size(), b.size()))
        n <<= 1;
    n <<= 1;
    fa.resize (n),  fb.resize (n);

    fft(fa, false),  fft(fb, false);
    for (size_t i = 0; i < n; ++i)
        fa[i] *= fb[i];
    fft(fa, true);

    res.resize (n);
    for (size_t i = 0; i < n; ++i) {
        // это будет работать неправильно для отрицательных коэффициентов в задаче перемножения многочленов
        // res[i] = int(fa[i].real() + 0.5);
        res[i] = (int)lround(fa[i].real());
    }

    int carry = 0;
    for (size_t i = 0; i < n; ++i) {
        res[i] += carry;
        carry = res[i] / 10;
        res[i] %= 10;
    }
}

void multiply_fft_parallel(const std::vector<int> & a, const std::vector<int> & b, std::vector<int> & res) {
    vcd fa (a.begin(), a.end()),  fb (b.begin(), b.end());
    size_t n = 1;
    while (n < std::max(a.size(), b.size()))
        n <<= 1;
    n <<= 1;
    fa.resize (n),  fb.resize (n);

    //fft_parallel(fa, false);
    //fft_parallel(fb, false);

    omp_set_nested(1);

#pragma omp parallel sections
    {
#pragma omp section
        {
            fft_parallel(fa, false);
        }
#pragma omp section
        {
            fft_parallel(fb, false);
        }
    }

    //omp_set_num_threads(1);
#pragma omp parallel shared(n, fa, fb)
    {
#pragma omp for schedule (static)

        for (size_t i = 0; i < n; ++i)
            fa[i] *= fb[i];

    }

    fft_parallel(fa, true);

    res.resize (n);

#pragma omp parallel shared(n, res)
    {
#pragma omp for schedule (static)

        for (size_t i = 0; i < n; ++i) {
            // это будет работать неправильно для отрицательных коэффициентов в задаче перемножения многочленов
            // res[i] = int(fa[i].real() + 0.5);
            res[i] = (int)lround(fa[i].real());
        }

    }

    int carry = 0;
    for (size_t i = 0; i < n; ++i) {
        res[i] += carry;
        carry = res[i] / 10;
        res[i] %= 10;
    }
}

void test_fft_multiply(std::string& s1, std::string& s2,
                       void (*f)(const std::vector<int> &, const std::vector<int> &, std::vector<int> &),
                       const std::string& name) {
    std::vector<int> p1;
    std::vector<int> p2;
    read_from_str_FFT(p1, s1);
    read_from_str_FFT(p2, s2);
    vcd p1_t(p1.begin(), p1.end());
    vcd p2_t(p2.begin(), p2.end());
    std::vector<int> res;

    auto begin = std::chrono::steady_clock::now();
    f(p1, p2, res);
    auto end = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
    std::cout << "test_" + name + "_multiply time: " << elapsed_ms.count() << " ms\n";

    //while (res.size() > 1 && res.back() == 0) res.pop_back();
    //for (auto it = res.rbegin(); it != res.rend(); ++it) { std::cout << *it; }
    //std::cout << std::endl;
}

void test_base_multiply(std::string& s1, std::string& s2) {
    lnum a, b;
    read_from_str_Base(a, s1);
    read_from_str_Base(b, s2);

    auto begin = std::chrono::steady_clock::now();
    auto c = multiply_brute(a, b);
    auto end = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
    std::cout << "test_base_multiply time: " << elapsed_ms.count() << " ms\n";

    //print_lnum(c);
}

void test_all() {
    //srand(time(nullptr));
    std::string s1(20000000, '6');
    std::string s2(15000000, '8');

    //test_base_multiply(s1, s2);
    //test_fft_multiply(s1, s2, multiply_FFT, "FFT");
    test_fft_multiply(s1, s2, multiply_fft, "fft");
    test_fft_multiply(s1, s2, multiply_fft_parallel, "fft_parallel");
}

int main(int argc, char *argv[]) {
    test_all();
    return 0;
}

// std::string s1(2000000, '7');
// std::string s2(2000000, '8');
// test_fft_multiply time: 12074 ms
// test_FFT_multiply time: 31787 ms
// test_base_multiply time: 1162196 ms


// std::string s1(20000000, '6');
// std::string s2(15000000, '8');
// test_fft_multiply time: 366041 ms
// test_fft_parallel_multiply time: 125878 ms
