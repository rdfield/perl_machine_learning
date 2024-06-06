use Modern::Perl;
use lib '.';
use ML::MNIST;
use Data::Dumper;
use File::Slurp;
use JSON;
use Math::Matrix;

use Math::Random;
use List::Util qw/shuffle/;
use Time::HiRes qw(gettimeofday tv_interval);
use ML::GPU qw(create_network print_network feedforward calculate_loss backprop update_weights SGD save_network);

my $debug = 0;
my $load_from_file = 0;
my $batch_size = 10;
my $learning_rate = 0.025;
my $weight_init = "scaled"; 
my $loss_function = "CrossEntropy"; # or MSE
my $epochs = 75;
my $sizes = [784, 30, 10];
my $lambda = 5;

package mnist_loader {
   use Python::Serialise::Pickle;
   use Data::Dumper;
   use lib '.';
   use ML::MNIST;
   sub load_data_wrapper {
       my $MNIST = ML::MNIST->new();
       my $train_data = $MNIST->load_train_data();
       my $train_labels = $MNIST->load_train_labels();
       my $test_data = $MNIST->load_test_data();
       my $test_labels = $MNIST->load_test_labels();
       my $training_data = [];
       my $validation_data = [];
       my $testing_data = [];
       foreach my $i (0 .. $#{$train_data}) {
          if ($i<50000) {
             push @$training_data, [ $train_data->[$i], $train_labels->[$i] ];
          } else {
             push @$validation_data, [ $train_data->[$i], $train_labels->[$i] ];
          }
       }
       foreach my $i (0 .. $#{$test_data}) {
          push @$testing_data, [ $test_data->[$i], $test_labels->[$i] ];
       }
        
       return ($training_data, $validation_data, $testing_data);
   }
};

$|++;

my ($training_data, $validation_data, $test_data) = mnist_loader::load_data_wrapper();
say scalar(@$training_data) . " training items";
say scalar(@$validation_data) . " validation items";
say scalar(@$test_data) . " test items";


#my $biases = from_json(scalar(read_file("bias.json")));
#my $weights = from_json(scalar(read_file("weights.json")));
my $biases;
my $weights;

my $net = create_network(
                         $sizes,  
                         bias => $biases, 
                         weights => $weights, 
                         debug => $debug, 
                         batch_size => $batch_size, 
                         loss => $loss_function,
                         weight_init => $weight_init
                         );

SGD($training_data, $epochs, $batch_size, $learning_rate, test_data=>$test_data, load_from_file => $load_from_file, lambda => $lambda,
    evaluation_data => $validation_data,
    monitor_training_cost => 1, monitor_evaluation_cost => 1,
    monitor_training_accuracy => 1, monitor_evaluation_accuracy => 1
   );

save_network("test_network_1.json");
