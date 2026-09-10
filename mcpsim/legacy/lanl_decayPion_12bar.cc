// Dalitz decay of pi0/eta to a millicharged pair, LANSCE 12-bar geometry.
// Derived from Insung Hwang's lanl/decayPion.cc:
//   https://github.com/insungg/mcp-production/blob/master/lanl/decayPion.cc
// Geometry parametrized for mcpsim (bar array, configurable face).

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "TFile.h"
#include "TLorentzVector.h"
#include "TRandom3.h"
#include "TTree.h"

using namespace std;

const double defaultMotherMass = 0.1349768; // GeV, pi0
const double alpha = 0.0072973526; // 1/137
const double PI = 3.141592653589793;

struct Particle {
    TLorentzVector momentum;
};

double lambda(double M, double m1, double m2) {
    double kallen = pow(M, 4) + pow(m1, 4) + pow(m2, 4)
        - 2 * pow(M * m1, 2)
        - 2 * pow(m1 * m2, 2)
        - 2 * pow(m2 * M, 2);
    return sqrt(max(0.0, kallen)) / (2.0 * M);
}

double dalitzWeight(double s, double theta, double mchi, double motherMass) {
    if (s <= 4.0 * mchi * mchi || s >= motherMass * motherMass) {
        return 0.0;
    }
    double xi = 1.0 - 4.0 * mchi * mchi / s;
    if (xi <= 0.0) {
        return 0.0;
    }
    return sin(theta)
        * alpha / (4.0 * PI * s)
        * pow(1.0 - s / (motherMass * motherMass), 3)
        * sqrt(xi) * (2.0 - xi * pow(sin(theta), 2));
}

double paperDalitzWeight(double s, double theta, double mchi, double motherMass) {
    if (s <= 4.0 * mchi * mchi || s >= motherMass * motherMass) {
        return 0.0;
    }
    double xi = 1.0 - 4.0 * mchi * mchi / s;
    if (xi <= 0.0) {
        return 0.0;
    }
    return sin(theta)
        * alpha / (4.0 * PI * s)
        * pow(1.0 - s / (motherMass * motherMass), 3)
        * sqrt(xi) * xi * pow(sin(theta), 2);
}

double angularMax(double xi) {
    if (xi <= 0.0) {
        return 0.0;
    }

    // Maximize u * (2 - xi*u^2), where u = sin(theta) in [0, 1].
    if (xi < 2.0 / 3.0) {
        return 2.0 - xi;
    }

    double u = sqrt(2.0 / (3.0 * xi));
    return u * (2.0 - xi * u * u);
}

double estimateDalitzEnvelope(double mchi, double motherMass) {
    const double sMin = 4.0 * mchi * mchi;
    const double sMax = motherMass * motherMass;
    const int nGrid = 20000;
    double logMin = log(sMin * (1.0 + 1.0e-12));
    double logMax = log(sMax * (1.0 - 1.0e-12));
    double maxWeight = 0.0;

    for (int i = 0; i < nGrid; ++i) {
        double t = (i + 0.5) / nGrid;
        double s = exp(logMin + t * (logMax - logMin));
        double xi = 1.0 - 4.0 * mchi * mchi / s;
        if (xi <= 0.0) {
            continue;
        }
        double weight = alpha / (4.0 * PI * s)
            * pow(1.0 - s / (motherMass * motherMass), 3)
            * sqrt(xi)
            * angularMax(xi);
        weight *= s;
        maxWeight = max(maxWeight, weight);
    }

    return 1.10 * maxWeight;
}

class LegacyCdfSampler {
public:
    LegacyCdfSampler(double mchiIn, double motherMassIn)
        : mchi(mchiIn), motherMass(motherMassIn) {
        build();
    }

    bool valid() const {
        return cdf.size() > 1 && cdf.back() > 0.0;
    }

    double sampleS(TRandom3& rnd) const {
        double target = rnd.Uniform(0.0, cdf.back());
        auto it = lower_bound(cdf.begin(), cdf.end(), target);
        if (it == cdf.begin()) {
            return exp(logS.front());
        }
        if (it == cdf.end()) {
            return exp(logS.back());
        }

        size_t hi = static_cast<size_t>(it - cdf.begin());
        size_t lo = hi - 1;
        double denom = cdf[hi] - cdf[lo];
        double frac = (denom > 0.0) ? (target - cdf[lo]) / denom : 0.0;
        return exp(logS[lo] + frac * (logS[hi] - logS[lo]));
    }

    double sampleTheta(double s, TRandom3& rnd) const {
        double xi = max(0.0, min(1.0, 1.0 - 4.0 * mchi * mchi / s));
        while (true) {
            double u = rnd.Uniform(-1.0, 1.0);
            double weight = 2.0 - xi + xi * u * u;
            if (rnd.Uniform(0.0, 2.0) <= weight) {
                return acos(max(-1.0, min(1.0, u)));
            }
        }
    }

private:
    double mchi;
    double motherMass;
    vector<double> logS;
    vector<double> cdf;

    double marginalLogSWeight(double s) const {
        double xi = 1.0 - 4.0 * mchi * mchi / s;
        if (xi <= 0.0 || s >= motherMass * motherMass) {
            return 0.0;
        }
        const double legacyBrPi2gg = 1.0e9;
        return alpha / (4.0 * PI)
            * pow(1.0 - s / (motherMass * motherMass), 3)
            * sqrt(xi)
            * 4.0 * (1.0 - xi / 3.0)
            * legacyBrPi2gg;
    }

    void build() {
        const int nGrid = 20000;
        double tMin = log(4.0 * mchi * mchi * (1.0 + 1.0e-12));
        double tMax = log(motherMass * motherMass * (1.0 - 1.0e-12));
        if (!isfinite(tMin) || !isfinite(tMax) || tMin >= tMax) {
            return;
        }

        logS.resize(nGrid);
        cdf.resize(nGrid);
        double previousWeight = 0.0;
        for (int i = 0; i < nGrid; ++i) {
            double frac = static_cast<double>(i) / static_cast<double>(nGrid - 1);
            double t = tMin + frac * (tMax - tMin);
            double weight = marginalLogSWeight(exp(t));
            logS[i] = t;

            if (i == 0) {
                cdf[i] = 0.0;
            } else {
                double dt = logS[i] - logS[i - 1];
                cdf[i] = cdf[i - 1] + 0.5 * (previousWeight + weight) * dt;
            }
            previousWeight = weight;
        }
    }
};

bool hitsRectangularDetector(const TLorentzVector& mom, double distanceM, double widthM, double heightM) {
    if (mom.Px() <= 0.0) {
        return false;
    }

    // LANSCE-mQ ER1/ER2 are off-axis from the proton beamline.
    // With the beam along z, model the detector as a side plane x = distanceM,
    // with its rectangular face spanning y and z.
    double y = distanceM * mom.Py() / mom.Px();
    double z = distanceM * mom.Pz() / mom.Px();
    return fabs(y) <= 0.5 * widthM && fabs(z) <= 0.5 * heightM;
}

int main(int argc, char** argv) {
    if (argc < 5 || argc > 14) {
        cerr << "Usage: " << argv[0]
             << " <output_root> <mchi_GeV> <eff_output_txt> <meson_text_file>"
             << " [distance_m=6.0] [bar_columns=3] [bar_rows=2]"
             << " [mother_mass_GeV=0.1349768] [random_seed=12345]"
             << " [sampler_mode=dalitz] [write_root=1] [bar_layers=2]"
             << " [detector_face_mode=rectangular]\n"
             << "sampler_mode: dalitz, legacy_placeholder, legacy_rejection, or paper_rejection\n";
        return 1;
    }

    const string outRoot = argv[1];
    const double mchi = atof(argv[2]);
    const string outEffFile = argv[3];
    const string infile = argv[4];
    const double distanceM = (argc > 5) ? atof(argv[5]) : 6.0;
    const int barColumns = (argc > 6) ? atoi(argv[6]) : 3;
    const int barRows = (argc > 7) ? atoi(argv[7]) : 2;
    const double motherMass = (argc > 8) ? atof(argv[8]) : defaultMotherMass;
    const unsigned int randomSeed = (argc > 9) ? static_cast<unsigned int>(strtoul(argv[9], nullptr, 10)) : 12345;
    const string samplerMode = (argc > 10) ? argv[10] : "dalitz";
    const bool writeRoot = (argc > 11) ? (atoi(argv[11]) != 0) : true;
    const int barLayers = (argc > 12) ? atoi(argv[12]) : 2;
    const string detectorFaceMode = (argc > 13) ? argv[13] : "rectangular";
    const double barSizeM = 0.05;
    const double frontFaceAreaM2 = barColumns * barRows * barSizeM * barSizeM;
    double detectorWidthM = barColumns * barSizeM;
    double detectorHeightM = barRows * barSizeM;
    if (detectorFaceMode == "square_area" || detectorFaceMode == "square") {
        const double sideM = sqrt(frontFaceAreaM2);
        detectorWidthM = sideM;
        detectorHeightM = sideM;
    } else if (detectorFaceMode != "rectangular") {
        cerr << "Unknown detector_face_mode=" << detectorFaceMode
             << ". Use rectangular or square_area.\n";
        return 3;
    }

    if (mchi <= 0.0 || motherMass <= 0.0 || distanceM <= 0.0
        || barColumns <= 0 || barRows <= 0 || barLayers <= 0) {
        cerr << "Inputs must be positive: mchi, mother mass, distance, bar columns, bar rows, and bar layers.\n";
        return 2;
    }
    if (2.0 * mchi >= motherMass) {
        cerr << "Illegal Dalitz mass: 2*mchi >= mother mass for mchi=" << mchi
             << ", motherMass=" << motherMass << "\n";
        return 4;
    }
    if (samplerMode != "dalitz" && samplerMode != "legacy_placeholder"
        && samplerMode != "legacy_rejection" && samplerMode != "paper_rejection") {
        cerr << "Unknown sampler mode: " << samplerMode
             << ". Use dalitz, legacy_placeholder, legacy_rejection, or paper_rejection.\n";
        return 5;
    }

    const double dalitzEnvelope = (samplerMode == "dalitz")
        ? estimateDalitzEnvelope(mchi, motherMass)
        : numeric_limits<double>::quiet_NaN();
    if (samplerMode == "dalitz" && (!isfinite(dalitzEnvelope) || dalitzEnvelope <= 0.0)) {
        cerr << "Failed to estimate Dalitz rejection-sampling envelope.\n";
        return 6;
    }

    LegacyCdfSampler legacySampler(mchi, motherMass);
    if (samplerMode == "legacy_placeholder" && !legacySampler.valid()) {
        cerr << "Failed to build legacy 1e9 CDF sampler.\n";
        return 6;
    }

    ifstream in(infile);
    if (!in) {
        cerr << "Failed to open meson data " << infile << "\n";
        return 7;
    }

    TFile* fout = nullptr;
    TTree* treeAll = nullptr;
    TTree* treeHit = nullptr;
    if (writeRoot) {
        fout = TFile::Open(outRoot.c_str(), "RECREATE");
        if (!fout || fout->IsZombie()) {
            cerr << "Failed to open output ROOT file " << outRoot << "\n";
            return 8;
        }

        treeAll = new TTree("mcp_all", "All MCPs");
        treeHit = new TTree("mcp_hit", "MCPs intersecting 12-bar detector");
    }

    double PX, PY, PZ, PP, M, PHI, THETA;
    auto branchify = [&](TTree* t) {
        t->Branch("Px", &PX);
        t->Branch("Py", &PY);
        t->Branch("Pz", &PZ);
        t->Branch("P", &PP);
        t->Branch("M", &M);
        t->Branch("Phi", &PHI);
        t->Branch("Theta", &THETA);
    };
    if (writeRoot) {
        branchify(treeAll);
        branchify(treeHit);
    }

    TRandom3 rnd(randomSeed);
    double legacyEnvelope = 150.0;

    double pMeV, cosTheta, phiLab;
    string line;
    long long mesonCount = 0;
    long long totalCount = 0;
    long long hitCount = 0;
    while (getline(in, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        istringstream row(line);
        if (!(row >> pMeV >> cosTheta >> phiLab)) {
            continue;
        }

        cosTheta = max(-1.0, min(1.0, cosTheta));
        double pLab = pMeV / 1000.0;
        if (pLab <= 0.0) {
            continue;
        }

        double thetaLab = acos(cosTheta);
        ++mesonCount;

        double s, thetaRel;
        while (true) {
            if (samplerMode == "legacy_placeholder") {
                s = legacySampler.sampleS(rnd);
                thetaRel = legacySampler.sampleTheta(s, rnd);
                break;
            }

            if (samplerMode == "dalitz") {
                double logSMin = log(4.0 * mchi * mchi);
                double logSMax = log(motherMass * motherMass);
                s = exp(rnd.Uniform(logSMin, logSMax));
            } else {
                s = rnd.Uniform(4.0 * mchi * mchi, motherMass * motherMass);
            }
            thetaRel = rnd.Uniform(0.0, PI);
            double y = (samplerMode == "paper_rejection")
                ? paperDalitzWeight(s, thetaRel, mchi, motherMass)
                : dalitzWeight(s, thetaRel, mchi, motherMass);
            double envelope = dalitzEnvelope;
            if (samplerMode == "legacy_rejection" || samplerMode == "paper_rejection") {
                y *= 1.0e9;
                envelope = legacyEnvelope;
            } else {
                y *= s;
            }
            if (samplerMode == "dalitz" && y > envelope) {
                cerr << "Dalitz envelope underestimated for mchi=" << mchi
                     << ": y=" << y << ", envelope=" << envelope << "\n";
                return 9;
            }
            if (y >= rnd.Uniform(0.0, envelope)) {
                if ((samplerMode == "legacy_rejection" || samplerMode == "paper_rejection") && y > legacyEnvelope) {
                    legacyEnvelope = y;
                }
                break;
            }
        }

        double pDecay = lambda(sqrt(s), mchi, mchi);
        double phiDecay = rnd.Uniform(0.0, 2.0 * PI);

        Particle chi1, chi2;
        chi1.momentum.SetPxPyPzE(
            pDecay * sin(thetaRel) * cos(phiDecay),
            pDecay * sin(thetaRel) * sin(phiDecay),
            pDecay * cos(thetaRel),
            sqrt(pDecay * pDecay + mchi * mchi)
        );
        chi2.momentum.SetPxPyPzE(
            -chi1.momentum.Px(),
            -chi1.momentum.Py(),
            -chi1.momentum.Pz(),
            chi1.momentum.E()
        );

        double vP = lambda(motherMass, sqrt(s), 0.0);
        double vBeta = vP / sqrt(vP * vP + s);
        double vTheta = rnd.Uniform(0.0, PI);
        double vPhi = rnd.Uniform(0.0, 2.0 * PI);
        if (samplerMode == "paper_rejection") {
            chi1.momentum.Boost(0.0, 0.0, vBeta);
            chi2.momentum.Boost(0.0, 0.0, vBeta);
            chi1.momentum.RotateZ(vTheta);
            chi1.momentum.RotateY(vPhi);
            chi2.momentum.RotateZ(vTheta);
            chi2.momentum.RotateY(vPhi);
        } else {
            double bvx = vBeta * sin(vTheta) * cos(vPhi);
            double bvy = vBeta * sin(vTheta) * sin(vPhi);
            double bvz = vBeta * cos(vTheta);
            chi1.momentum.Boost(bvx, bvy, bvz);
            chi2.momentum.Boost(bvx, bvy, bvz);
        }

        double ePi = sqrt(pLab * pLab + motherMass * motherMass);
        double bx = pLab * sin(thetaLab) * cos(phiLab) / ePi;
        double by = pLab * sin(thetaLab) * sin(phiLab) / ePi;
        double bz = pLab * cos(thetaLab) / ePi;
        chi1.momentum.Boost(bx, by, bz);
        chi2.momentum.Boost(bx, by, bz);

        for (Particle* chi : {&chi1, &chi2}) {
            const TLorentzVector& mom = chi->momentum;
            PX = mom.Px();
            PY = mom.Py();
            PZ = mom.Pz();
            PP = mom.P();
            M = mom.M();
            PHI = mom.Phi();
            THETA = mom.Theta();

            ++totalCount;
            if (writeRoot) {
                treeAll->Fill();
            }
            bool hit = hitsRectangularDetector(mom, distanceM, detectorWidthM, detectorHeightM);
            if (hit) {
                ++hitCount;
                if (writeRoot) {
                    treeHit->Fill();
                }
            }
        }
    }
    in.close();

    double total = totalCount;
    double hits = hitCount;
    double efficiency = (total > 0.0) ? hits / total : 0.0;

    if (writeRoot) {
        fout->Write();
        fout->Close();
    }

    ofstream effOut(outEffFile);
    if (!effOut.is_open()) {
        cerr << "Failed to open efficiency output " << outEffFile << "\n";
        return 10;
    }
    effOut << setprecision(10)
           << mchi << " "
           << efficiency << " "
           << static_cast<long long>(hits) << " "
           << static_cast<long long>(total) << " "
           << distanceM << " "
           << detectorWidthM << " "
           << detectorHeightM << " "
           << motherMass << " "
           << barColumns << " "
           << barRows << " "
           << barLayers << "\n";
    effOut.close();

    cout << "Processed " << mesonCount << " mesons for mchi=" << mchi << " GeV\n"
         << "Detector: " << barColumns << " x " << barRows
         << " front-face bars x " << barLayers << " depth layers"
         << ", " << detectorFaceMode << " accepted face with area=" << frontFaceAreaM2 << " m^2"
         << ", width=" << detectorWidthM << " m, height=" << detectorHeightM
         << " m, side-plane distance=" << distanceM << " m, motherMass=" << motherMass << " GeV\n"
         << "samplerMode=" << samplerMode
         << ", dalitzEnvelope=" << dalitzEnvelope
         << ", legacyEnvelope=" << legacyEnvelope
         << ", randomSeed=" << randomSeed
         << ", writeRoot=" << writeRoot << "\n"
         << "Accepted MCPs: " << hits << " / " << total
         << " ; efficiency=" << efficiency << "\n";

    return 0;
}
