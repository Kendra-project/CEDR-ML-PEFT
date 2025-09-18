int predict(float feature[12]) {
    int label;
    if ( feature[1] <= 38.500000 ) {
        label = 4;
    } else {
        if ( feature[3] <= 3019.000000 ) {
            label = 3;
        } else {
            label = 2;
        }
    }
    return label;
}
