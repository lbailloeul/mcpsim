// This Pythia script generates p-p fixed target collision events
// and stoe momentum of pi0, eta, J/psi and upsilon
// Author : Insung Hwang (his5624@korea.ac.kr)
// Editor: Leo Bailloeul (lbailloeul@gmail.com)
#include <iostream>
#include <string>
#include <stdlib.h>
#include <unistd.h>
#include <cmath>
#include "TFile.h"
#include "TTree.h"
#include "TThread.h"
#include "Pythia8/Pythia.h"

using namespace std;

// multi-threading callback function
void* handler(void *t_argv);


// usage : program_name [seed] [nCores] [nEvents]
int main(int argc, char *argv[])
{
    if (argc < 5) {
        cout << "REQUIRED FORMAT : submet [seed] [nCores] [nEvents]" << endl;
        return -1;
    }

    int seed = strtol(argv[1], NULL, 10);
    int nCores = strtol(argv[2], NULL, 10);
    int nEvents = strtol(argv[3], NULL, 10);
	int idNumber = strtol(argv[4], NULL, 10);
    int nJobs = nEvents / nCores;
    cout << "seed : " << seed << "    nCores : " << nCores << "    nJobs : " << nJobs << endl;

    // define threads for multi-threading
    TThread* th[nCores];
    int thread_argv[4];
    thread_argv[0] = seed;
    thread_argv[1] = nJobs;
	thread_argv[3] = idNumber;

    // throw jobs to threads
    for (int i = 0; i < nCores; i++) {
        thread_argv[2] = i;
        th[i] = new TThread(Form("th%d", i), handler, (void *) thread_argv);
        th[i]->Run();
	    sleep(1); // wait until callbalck initialization finish
    }

    // join works when each josbs are finished
    for(int i = 0; i < nCores; i++)
        th[i]->Join();
    return 0;
}

void* handler(void *t_argv)
{
    int *parameters = (int *) t_argv;

    int seed = parameters[0];
    int nJobs = parameters[1];
    int ith = parameters[2];
	int idNumber = parameters[3];

    // define root file
    TFile* output = new TFile( Form( "../output-data/mesons_seed%d_t%d.root", seed, ith) , "RECREATE" );

    // define kinematic variables
    int id;
    double px, py, pz, pt, p, M, e, mag, phi, theta;
	double x, y, z;

    // create branches
    TTree *tree = new TTree("mesons", "mesons");
    tree->Branch("id", &id);
    tree->Branch("px", &px);
    tree->Branch("py", &py);
    tree->Branch("pz", &pz);
    tree->Branch("mass",   &M);
	tree->Branch("phi", &phi);
	tree->Branch("theta", &theta);
	tree->Branch("e",   &e);
	tree->Branch("magnitude",   &mag);


    // create pythia generator
    Pythia8::Pythia pythia;

    // read beam properties
    pythia.readFile("beam.config");
    pythia.readFile("momentum.config");

    // define random seed
    pythia.readString("Random:setSeed = on"); // use random seed
    pythia.readString(Form("Random:seed = %d", seed + ith)); // + ith to prevent redundant event generation

    pythia.init();


    // pythia.particleData.list(31); // check if mcp is defined

	int mesonID[1] = {idNumber};// {pi0:111, eta:221, J/psi:443, upsilon:553, rho:113, omega:223, phi:333};
    for (int i = 0; i < nJobs; i ++) {
        if (!pythia.next()) continue; // skip when generation failed

		for (int j = 0; j < pythia.event.size(); j++) {
			for (int k = 0; k < 1; k++) {
				if ( abs(pythia.event.at(j).id()) == mesonID[k]) {
					id = pythia.event.at(j).id();
					px = pythia.event.at(j).px();
					py = pythia.event.at(j).py();
					pz = pythia.event.at(j).pz();
					M = pythia.event.at(j).m();
					phi = pythia.event.at(j).phi();
					theta = pythia.event.at(j).theta();
					e = pythia.event.at(j).e();
					mag = sqrt(px*px + py*py + pz*pz);
					tree->Fill();

				}
			}
		}
    }
    output->Write();
    output->Close();
}