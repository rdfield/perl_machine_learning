se Modern::Perl;
use lib '.';
use ML::MNIST;
use Data::Dumper;
use File::Slurp;
use JSON;
use Math::Matrix;
use Math::Random;
use List::Util qw/shuffle/;
use Time::HiRes qw(gettimeofday tv_interval);
use ML::ROCM qw(load_network mnist_image_guess);

load_network("rocm_network_1.json");

my $image = scalar(read_file(shift));

my @btmp = unpack('C*',$image);
foreach my $b (0 .. $#btmp) {
   $btmp[$b] = $btmp[$b]/255;
}
say join(",",@{mnist_image_guess(\@btmp)});
