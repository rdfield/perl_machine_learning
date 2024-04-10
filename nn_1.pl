# This is a Perl implementation of the Python code presented in http://neuralnetworksanddeeplearning.com/chap1.html
# It is extemely slow, taking 40 minutes to complete a single epoch.  About 95% accurate.
# If set, the "load_from_file" option will load initial weights, biases and mini batches from JSON files.  
# To create these JSON files, edit the code in http://neuralnetworksanddeeplearning.com/chap1.html to output JSON
# rather than pickle files.  There is a version of this code in this repo: nn_1.py.
# The code is self contained, with a couple of obvious exceptions: ML::MNIST is in this repository, and the rest
# of the packages are either distributed with Perl or available via CPAN.
use Modern::Perl;
use lib '.';
use ML::MNIST;
use Data::Dumper;
use File::Slurp;
use JSON;
use Math::Matrix;

package Network {

use Data::Dumper;
use Math::Random;
use File::Slurp;
use JSON;
use List::Util qw/shuffle/;

   sub new {
      my $proto = shift;
      my $class = ref($proto) || $proto;

      my $sizes = shift;
      my $num_layers = scalar(@$sizes);
      my %params = @_;

      my $biases = [];
      my $weights = [];
      my $mb;
      if ($params{load_from_file}) {
         $biases = from_json(scalar(read_file("bias.json")));
         foreach my $b (@$biases) {
            $b = Math::Matrix->new($b);
         }
         $weights = from_json(scalar(read_file("weights.json")));
         foreach my $w (@$weights) {
            $w = Math::Matrix->new($w);
         }
      } else {
         foreach my $s (1 .. $#{$sizes}) {
            push @$biases, Math::Matrix->new([random_normal($sizes->[$s])])->transpose();
         }
         foreach my $s (0 .. ($#{$sizes} - 1)) {
            my $iw = [];
            foreach my $x (0 .. ($sizes->[$s + 1] - 1)) {
               push @$iw, [random_normal( $sizes->[$s])];
            }
            push @$weights, Math::Matrix->new($iw);
         }
      }
      my $self = { num_layers => $num_layers, sizes => $sizes, biases => $biases, weights => $weights, load_from_file => $params{load_from_file} };
      return bless( $self, $class );
   }
   
   sub sizes {
      my $self = shift;
      return $self->{sizes};
   }
  
   sub mb {
      my $self = shift;
      return $self->{mb};
   }

   sub num_layers {
      my $self = shift;
      return $self->{num_layers};
   }

   sub load_from_file {
      my $self = shift;
      return $self->{load_from_file};
   }
  
   sub biases {
      my $self = shift;
      return $self->{biases};
   }
  
   sub weights {
      my $self = shift;
      return $self->{weights};
   }

   sub sigmoid {
      my $self = shift;
      my $z = shift;
      return $z->map( sub { $_ = 1/( 1 + exp( -1 * $_) ) });
   }

   sub sigmoid_prime {
      my $self = shift;
      my $z = shift;
      my $sigz = $self->sigmoid($z);
      return $sigz->map( sub { $_*(1-$sigz->[ $_[0] ][ $_[1] ] )}); # element multiplication
   }
  
   sub cost_derivative {
      my ($self, $output_activations, $y) = @_;
      my $cd = $output_activations->map( sub { $_ - $y->[$_[0]][$_[1]] } );
      return $cd;
   }

   sub backprop {
       my $self = shift;
       my ($x, $y) = @_;
       my $nabla_b = [];
       my $nabla_w = [];
       foreach my $s (1 .. $#{$self->sizes}) {
          push @$nabla_b, [([0]) x $self->sizes->[$s]];
       }
       foreach my $s (0 .. ($#{$self->sizes} - 1)) {
          my $iw = [];
          foreach my $x (0 .. ($self->sizes->[$s + 1] - 1)) {
             push @$iw, [([0]) x $self->sizes->[$s]];
          }
          push @$nabla_w, $iw;
       }
       my $activation = Math::Matrix->new($x);
       my @activations = ( $activation );
       my @zs = ();
       foreach my $i (0 .. $#{$self->biases}) {
          my $b = $self->biases->[$i];
          my $w = $self->weights->[$i];
          my $z = ($w * $activation)->map( sub { $_ += $b->[ $_[0] ][ $_[1] ] } );
          push @zs, $z;
          $activation = $self->sigmoid($z);
          push @activations, $activation;
       }
       my $spzs = $self->sigmoid_prime($zs[-1]);
       my $delta = $self->cost_derivative($activations[-1], $y)->map(  sub { $_ * $spzs->[$_[0]][$_[1]] } );
       $nabla_b->[-1] = $delta;
       $nabla_w->[-1] = $delta * $activations[-2]->transpose();
       foreach my $l (2 .. ($self->num_layers - 1)) {
            my $z = $zs[-$l];
            my $sp = $self->sigmoid_prime($z);
            my $delta = ($self->weights()->[-$l+1]->transpose() *  $delta)->map( sub { $_ * $sp->[ $_[0] ][ $_[1] ] } );
            $nabla_b->[-$l] = $delta;
            $nabla_w->[-$l] = $delta * $activations[-$l-1]->transpose();
        }
       return ($nabla_b, $nabla_w);
   }
   
   sub update_mini_batch {
       my $self = shift;
       my ($mini_batch, $eta, $epoch, $batchno) = @_;
       if ($self->load_from_file) {
          my $file_name = sprintf("mini_batch_e%04i_b%05i",($epoch - 1),$batchno);
          $mini_batch = from_json(scalar(read_file($file_name)));
       }
       my $nabla_b = [];
       my $nabla_w = [];
       foreach my $s (0 .. ($#{$self->sizes} - 1)) {
          foreach my $w (0 .. ($self->sizes->[$s + 1] - 1)) {
             $nabla_b->[$s][$w][0] = 0;
          } 
          $nabla_b->[$s] = Math::Matrix->new($nabla_b->[$s]);
       }
       foreach my $s (0 .. ($#{$self->sizes} - 1)) {
          foreach my $x (0 .. ($self->sizes->[$s + 1] - 1)) {
             foreach my $y (0 .. ($self->sizes->[$s] - 1)) {
                $nabla_w->[$s][$x][$y] = 0;
             }
          }
          $nabla_w->[$s] = Math::Matrix->new($nabla_w->[$s]);
       }
       my $modifier = $eta/(scalar(@$mini_batch));
       my $ctr = 0;
       foreach my $mb (@$mini_batch) {
          $ctr++;
          my $x = $mb->[0];
          my $y = $mb->[1];
          my ($delta_nabla_b, $delta_nabla_w) = $self->backprop($x, $y);
          foreach my $i ( 0 .. $#$nabla_b ) {
             $nabla_b->[$i] = $nabla_b->[$i]->map( sub { $_ + $delta_nabla_b->[$i]->[ $_[0] ][ $_[1] ] } ); 
          }
          foreach my $i ( 0 .. $#$nabla_w ) {
             $nabla_w->[$i] = $nabla_w->[$i]->map( sub { $_ + $delta_nabla_w->[$i]->[ $_[0] ][ $_[1] ] } ); 
          }
      }
      foreach my $i (0 .. $#$nabla_b) {
         $self->{biases}[$i] = $self->{biases}[$i]->map( sub { $_ - $modifier * ($nabla_b->[$i]->[$_[0]][$_[1]] // 0)} );
      }
      foreach my $i (0 .. $#$nabla_w) {
         $self->{weights}[$i] = $self->{weights}[$i]->map( sub { $_ - $modifier * $nabla_w->[$i]->[$_[0]][$_[1]] } );
      }
   }

   sub feedforward {
      my $self = shift;
      my $a = Math::Matrix->new(shift);
      foreach my $i (0 .. $#{$self->biases}) {
         my $b = $self->biases->[$i];
         my $w = $self->weights->[$i];
         $a = $self->sigmoid( ($w * $a)->map( sub { $_ + $b->[ $_[0] ] [ $_[1] ] } ) );
      }
      return $a;
   }

   sub argmax {
      my $self = shift;
      my $arr = shift;
      my $max = $arr->[0][0];
      my $idx = [0,0];
      foreach my $i (0 .. $#{$arr}) {
         foreach my $j (0 .. $#{$arr->[0]}) {
            if ($arr->[$i][$j] > $max) {
               $max = $arr->[$i][$j];
               $idx = [$i, $j];
            }
         }
      }
      return $idx;
   }

   sub evaluate {
      my $self = shift;
      my $data = shift;
      my $successes = 0;
      foreach my $d (@$data) {
         my $calc = $self->feedforward($d->[0]);
         if (ref($d->[1]) ne "Math::Matrix") {
           $d->[1] = Math::Matrix->new($d->[1]);
         }
         my $expected = $self->argmax($d->[1]);
         my $calculated = $self->argmax($calc);
         if ($expected->[0] == $calculated->[0] and $expected->[1] == $calculated->[1]) {
            $successes++;
         }
      }
      return $successes;
   }

   sub SGD {
      my $self = shift;
      my $training_data = shift;
      my $epochs = shift;
      my $mini_batch_size = shift;
      my $eta = shift;
      my %params = @_;
      my $test_data = $params{test_data};
      my $n = scalar(@$training_data);
      my $n_test;
      if ($test_data) {
         $n_test = scalar(@$test_data);
      }
      foreach my $j (1 .. $epochs) {
         my @training_data = shuffle(@$training_data);
         my @mini_batches; 
         my $k = 0;
         while (scalar(@training_data)) {
            push @mini_batches, [ splice @training_data, 0 , $mini_batch_size ];
         }
         my $ctr = 1;
         foreach my $mb (@mini_batches) {
            if ($ctr % 50 == 0) { print "." };
            $self->update_mini_batch($mb, $eta, $j, $ctr);
            $ctr++;
         }
         print "\n";
         if ($test_data) {
            say "Epoch $j : " . $self->evaluate($test_data) . " / $n_test";
         } else {
            say "Epoch $j complete";
         }
      }
   }
};

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
my $net = Network->new([784, 30, 10], load_from_file => 0);
$net->SGD($training_data, 30, 10, 3.0, test_data=>$test_data);
