// Burman-Smith pi0/eta generator for the 800 MeV LANSCE beam.
// Derived from Insung Hwang's lanl/generatePion.cc:
//   https://github.com/insungg/mcp-production/blob/master/lanl/generatePion.cc

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include "TRandom3.h"

using namespace std;

// Constants
const double defaultMesonMassMeV = 134.9768; // MeV, neutral pion mass
const double PI = 3.14159265358979;
const double conv2rad = PI / 180.0;

// Beam property
const int Z = 6;             // Carbon target
const double Tp = 800.0;     // MeV proton beam kinetic energy for LANSCE

// Distribution parameters
const int DEFAULT_TRIALS = 10000000;
const double knots[] = {0, 0, 0, 30, 70, 180, 180, 180};
const double B = 25.0;

double envelope = 10.0;
double TA, sigmaA, NormZ;
double bsplineCoeff[5];

void initialize();
double Bspline(int idx, int order, double theta);
double ddSigma(double Tpi, double theta);

int main(int argc, char* argv[]) {
    const string outputPath = (argc > 1) ? argv[1] : "Pion_Production.txt";
    const int trials = (argc > 2) ? atoi(argv[2]) : DEFAULT_TRIALS;
    const int seed = (argc > 3) ? atoi(argv[3]) : 12345;
    const double mesonMassMeV = (argc > 4) ? atof(argv[4]) : defaultMesonMassMeV;
    const double kineticMaxMeV = Tp - mesonMassMeV;

    if (mesonMassMeV <= 0.0 || kineticMaxMeV <= 0.0) {
        cerr << "Invalid meson mass " << mesonMassMeV
             << " MeV for " << Tp << " MeV proton kinetic energy.\n";
        return 2;
    }

    ofstream output(outputPath);
    if (!output.is_open()) {
        cerr << "Failed to create output file " << outputPath << endl;
        return 1;
    }

    initialize();
    TRandom3 rand(seed);

    output << setprecision(10);
    output << "# P_MeV cosTheta phi_rad\n";
    output << "# meson_mass_MeV " << mesonMassMeV << "\n";
    output << "# NOTE: angular/kinetic production shape is the LANL pi0 Burman-Smith model.\n";

    for (int i = 0; i < trials; i++) {
        while (true) {
            double Tpi = rand.Uniform(0.0, kineticMaxMeV);
            double thetaDeg = rand.Uniform(0.0, 180.0);
            double y = ddSigma(Tpi, thetaDeg);

            if (y >= rand.Uniform(0.0, 1.0) * envelope) {
                if (y > envelope) {
                    envelope = y;
                }

                double pMeV = sqrt(pow(Tpi + mesonMassMeV, 2) - pow(mesonMassMeV, 2));
                double thetaRad = thetaDeg * conv2rad;
                double phi = rand.Uniform(0.0, 2.0 * PI);

                output << pMeV << " " << cos(thetaRad) << " " << phi << "\n";
                break;
            }
        }

        if (i > 0 && i % 100000 == 0) {
            cout << i << " meson trials accepted" << endl;
        }
    }

    output.close();
    cout << "Wrote " << trials << " meson entries to " << outputPath
         << " with mass " << mesonMassMeV << " MeV" << endl;
    return 0;
}

void initialize() {
    double TA585, TA730, sigmaA585, sigmaA730;
    if (Z == 1) {
        TA730 = 53.0;
        TA585 = 64.5;
        sigmaA730 = 127.0;
        sigmaA585 = 155.0;
    } else if (Z <= 8) {
        TA730 = 34.2;
        TA585 = 28.9;
        sigmaA730 = 150.0;
        sigmaA585 = 130.0;
    } else if (Z < 92) {
        TA730 = 29.9;
        TA585 = 26.0;
        sigmaA730 = 166.0;
        sigmaA585 = 135.0;
    } else {
        cerr << "Z should be less than 93" << endl;
        exit(2);
    }

    TA = (TA730 * (Tp - 585) - TA585 * (Tp - 730)) / (730 - 585);
    sigmaA = (sigmaA730 * (Tp - 585) - sigmaA585 * (Tp - 730)) / (730 - 585);

    NormZ = 0;
    double normzc[] = {0.8851, -0.1015, 0.1459, -0.0265};
    for (int i = 0; i < 4; i++) {
        NormZ += normzc[i] * pow(log(Z), i) * pow(Z, 0.3333);
    }

    bsplineCoeff[0] = min(27.0 - 4.0 * pow((730.0 - Tp) / (730.0 - 585.0), 2), 27.0);
    bsplineCoeff[1] = 18.2;
    bsplineCoeff[2] = 8.0;
    bsplineCoeff[3] = 13.0 + (Z - 12.0) / 10.0;
    bsplineCoeff[4] = 9.0 + (Z - 12.0) / 10.0 - (Tp - 685.0) / 20.0;
}

double Bspline(int idx, int order, double theta) {
    if (order == 0) {
        return (theta >= knots[idx] && theta <= knots[idx + 1]) ? 1.0 : 0.0;
    }

    double total = 0;
    if (knots[idx + order] != knots[idx]) {
        total += (theta - knots[idx]) / (knots[idx + order] - knots[idx]) * Bspline(idx, order - 1, theta);
    }
    if (knots[idx + order + 1] != knots[idx + 1]) {
        total += (knots[idx + order + 1] - theta) / (knots[idx + order + 1] - knots[idx + 1]) * Bspline(idx + 1, order - 1, theta);
    }

    return total;
}

double ddSigma(double Tpi, double theta) {
    double Tbar = 48 + 330 * exp(-theta / TA);
    double sigma = sigmaA * exp(-theta / 85.0);
    double TF = Tp - 140 - 2 * B;

    double amp = 0;
    for (int i = 0; i < 5; i++) {
        amp += bsplineCoeff[i] * Bspline(i, 2, theta);
    }
    amp *= NormZ;

    return sin(theta * conv2rad) * amp
        * exp(-pow((Tbar - Tpi) / sqrt(2.0) / sigma, 2))
        / (1.0 + exp((Tpi - TF) / B));
}
