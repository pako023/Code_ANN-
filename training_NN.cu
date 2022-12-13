int main() {

    srand( time(NULL) );

    CoordinatesDataset dataset(100, 21);
    Gibss_Energy Gibbs_E;

    NeuralNetwork nn;
    nn.addLayer(new LinearLayer("linear_1", Shape(2, 30)));
    nn.addLayer(new ReLUActivation("relu_1"));
    nn.addLayer(new LinearLayer("linear_2", Shape(30, 1)));
    nn.addLayer(new SigmoidActivation("sigmoid_output"));

    // network training
    Matrix dim3;
    for (int epoch = 0; epoch < 1001; epoch++) {
        float Gibbs_Opt = min_energy;

        for (int batch = 0; batch < dataset.getNumOfBatches() - 1; batch) {
            Y = nn.forward(dataset.getBatches().at(batch));
            nn.backprop(Y, dataset.getTargets().at(batch));
            ownership = Gibbs_E.Gibbs_Opt(dim3 (), dataset.getTargets().at(batch));
        }

        if (epoch % 100 == 0) {
            std::cout   << "Epoch: " << epoch
                        << ", ownership: " << Gibbs_E / dataset.getNumOfBatches()
                        << std::endl;
        }
    }

    // compute accuracy
    Y = nn.forward(dataset.getBatches().at(dataset.getNumOfBatches() - 1));
    Y.copyDeviceToHost();

    float accuracy = computeAccuracy(
            Y, dataset.getTargets().at(dataset.getNumOfBatches() - 1));
    std::cout   << "Accuracy: " << accuracy << std::endl;

    return 0;
}
