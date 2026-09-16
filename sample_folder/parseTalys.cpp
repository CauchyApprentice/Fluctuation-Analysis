#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <limits>

struct LDPoint {
    double E;                 // excitation energy
    double aux;              // second column (often temperature / parameter)
    std::vector<double> rho; // partial level densities
};

class LDTable {
public:
    std::vector<LDPoint> data;

    // -----------------------------
    // Load TALYS LD table file
    // -----------------------------
    bool load(const std::string& filename) {
        std::ifstream fin(filename);
        if (!fin.is_open()) {
            std::cerr << "ERROR: cannot open file " << filename << std::endl;
            return false;
        }

        std::string line;
        while (std::getline(fin, line)) {

            if (line.empty()) continue;

            // skip comments
            if (line[0] == '#' || line[0] == '!' || line[0] == '%')
                continue;

            std::stringstream ss(line);
            std::vector<double> vals;
            double x;

            while (ss >> x) {
                vals.push_back(x);
            }

            // need at least: E + aux + 1 rho column
            if (vals.size() < 3) continue;

            LDPoint pt;
            pt.E = vals[0];
            pt.aux = vals[1];

            pt.rho.assign(vals.begin() + 2, vals.end());

            data.push_back(pt);
        }

        std::cout << "LDTable loaded: " << data.size() << " points\n";
        return !data.empty();
    }

    // -----------------------------
    // total level density at energy E
    // (log-linear interpolation)
    // -----------------------------
    double rho_total(double E) const {
        if (data.empty()) return 0.0;

        if (E <= data.front().E)
            return sum(data.front().rho);

        if (E >= data.back().E)
            return sum(data.back().rho);

        for (size_t i = 0; i < data.size() - 1; i++) {

            if (E >= data[i].E && E <= data[i + 1].E) {

                double E1 = data[i].E;
                double E2 = data[i + 1].E;

                double r1 = sum(data[i].rho);
                double r2 = sum(data[i + 1].rho);

                // avoid log(0)
                r1 = std::max(r1, 1e-300);
                r2 = std::max(r2, 1e-300);

                double ln1 = std::log(r1);
                double ln2 = std::log(r2);

                double t = (E - E1) / (E2 - E1);

                return std::exp(ln1 + t * (ln2 - ln1));
            }
        }

        return 0.0;
    }

    // -----------------------------
    // full component interpolation
    // -----------------------------
    std::vector<double> rho_components(double E) const {
        if (data.empty()) return {};

        if (E <= data.front().E)
            return data.front().rho;

        if (E >= data.back().E)
            return data.back().rho;

        for (size_t i = 0; i < data.size() - 1; i++) {

            if (E >= data[i].E && E <= data[i + 1].E) {

                const auto& A = data[i];
                const auto& B = data[i + 1];

                double t = (E - A.E) / (B.E - A.E);

                size_t n = A.rho.size();
                std::vector<double> out(n);

                for (size_t k = 0; k < n; k++) {

                    double a = std::max(A.rho[k], 1e-300);
                    double b = std::max(B.rho[k], 1e-300);

                    double la = std::log(a);
                    double lb = std::log(b);

                    out[k] = std::exp(la + t * (lb - la));
                }

                return out;
            }
        }

        return {};
    }

    // -----------------------------
    // utility: sum vector
    // -----------------------------
    static double sum(const std::vector<double>& v) {
        double s = 0.0;
        for (double x : v) s += x;
        return s;
    }
};

// -----------------------------
// Example usage
// -----------------------------
int main() {

    LDTable ld;

    if (!ld.load("your_talys_file.dat")) {
        return 1;
    }

    double E = 20.0;

    std::cout << "rho_total(" << E << ") = "
        << ld.rho_total(E) << std::endl;

    auto comp = ld.rho_components(E);

    std::cout << "components: ";
    for (double x : comp)
        std::cout << x << " ";

    std::cout << std::endl;

    return 0;
}