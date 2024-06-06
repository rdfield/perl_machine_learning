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
use ML::GPU qw(load_network mnist_batch_guess);


load_network("test_network_1.json");

my $batch = from_json(scalar(read_file("mini_batch_e0000_b00001_transposed")));

say Dumper(mnist_batch_guess($batch));
foreach my $r (@$batch) {
   say Dumper($r->[1]);
}
