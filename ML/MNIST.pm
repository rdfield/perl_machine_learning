# download the MNIST data from http://yann.lecun.com/exdb/mnist/ and uncomress the files into the $ENV{HOME}/MNIST directory
package ML::MNIST;
use Modern::Perl; 
use Data::Dumper;
use FindBin qw( $Bin );
use lib '.';

my $MNIST_TRAIN_DATA = "$Bin/MNIST/train-images-idx3-ubyte";
my $MNIST_TRAIN_LABELS = "$Bin/MNIST/train-labels-idx1-ubyte";
my $MNIST_TEST_DATA = "$Bin/MNIST/t10k-images-idx3-ubyte";
my $MNIST_TEST_LABELS = "$Bin/MNIST/t10k-labels-idx1-ubyte";

sub new {
   my $class = shift;
   my $proto = ref($class) || $class;
   my %params = @_;
   my $self = {};
   $self->{train_data} = $params{train_data} || $MNIST_TRAIN_DATA;
   $self->{train_labels} = $params{train_labels} || $MNIST_TRAIN_LABELS;
   $self->{test_data} = $params{test_data} || $MNIST_TEST_DATA;
   $self->{test_labels} = $params{test_labels} || $MNIST_TEST_LABELS;
   bless $self, $class;
   return $self;
}

sub normalise_data {
   my $data = shift;
   foreach my $i (0 .. $#$data) {
      $data->[$i][0] /= 255;
   }
}


sub load_train_data {
   my $self = shift;
   my %params = @_;
   $params{limit} ||= -1;
   open my $fh, '<', $self->{train_data}
      or die "Can't open file " . $self->{train_data} . ": $!";
   binmode($fh);
   my $data = _load_data($fh, $params{limit});
   #normalise_data($data);
   return $data;
}

sub load_test_data {
   my $self = shift;
   my %params = @_;
   $params{limit} ||= -1;
   open my $fh, '<', $self->{test_data}
      or die "Can't open file " . $self->{test_data} . ": $!";
   binmode($fh);
   my $data = _load_data($fh, $params{limit});
   #normalise_data($data);
   return $data;
}

sub load_train_labels {
   my $self = shift;
   my %params = @_;
   $params{limit} ||= -1;
   open my $fh, '<', $self->{train_labels}
      or die "Can't open file " . $self->{train_labels} . ": $!";
   binmode($fh);
   my $data = _load_labels($fh, $params{limit});
   my $l;
   foreach my $d (@$data) {
      my @row = ([0],[0],[0],[0],[0],[0],[0],[0],[0],[0]);
      $row[$d]->[0] = 1;
      push @$l, \@row;
   }
   return $l;
}

sub load_test_labels {
   my $self = shift;
   my %params = @_;
   $params{limit} ||= -1;
   open my $fh, '<', $self->{test_labels}
      or die "Can't open file " . $self->{test_labels} . ": $!";
   binmode($fh);
   my $data = _load_labels($fh, $params{limit});
   my $l;
   foreach my $d (@$data) {
      my @row = ([0],[0],[0],[0],[0],[0],[0],[0],[0],[0]);
      $row[$d]->[0] = 1;
      push @$l, \@row;
   }
   return $l;
}

sub _load_data {
   my $fh = shift;
   my $limit = shift;
   #Magic number
   my $buffer;
   read($fh, $buffer, 4);
   my $magic_number = unpack('N1', $buffer);
   if ($magic_number != 0x00000803) {
      die "Invalid magic number expected". 0x00000803. "actual $magic_number";
   }

   #Number of images
   read($fh, $buffer, 4);
   my $items = unpack('N1', $buffer);
   # Image row pixel count
   read($fh, $buffer, 4);
   my $rows = unpack('N1', $buffer);

   #Image column pixel count
   read($fh, $buffer, 4);
   my $cols = unpack('N1', $buffer);

   #Load images in array
   my $image;
   my $data = [];
   my $image_length = $rows * $cols;
   if ($limit == -1) {
      $limit = $items;
   }
   foreach my $i ( 1 .. $limit) {
      my $read = read $fh, $image, $image_length;
      unless ($read == $image_length) {
         die "Can't read image $i";
      }
      my @btmp = unpack('C*',$image);
      my @btmp2;
      foreach my $b (@btmp) {
         push @btmp2, [$b/255]; # the code in nn_1.pl requires the input data pre-transposed - see http://neuralnetworksanddeeplearning.com/chap2.html
                                # the divide by 255 is to standardise the data into the 0-1 range
      }
      push @$data, \@btmp2;
   }
   return $data;
}

sub _load_labels {
   my $fh = shift;
   my $limit = shift;
   #Magic number
   my $buffer;
   read($fh, $buffer, 4);
   my $magic_number = unpack('N1', $buffer);
   if ($magic_number != 0x00000801) {
      die "Invalid magic number expected". 0x00000801. "actual $magic_number";
   }

   #Number of images
   read($fh, $buffer, 4);
   my $items = unpack('N1', $buffer);
   my $labels;
   if ($limit == -1) {
      $limit = $items;
   }
   my $read = read $fh, $labels, $limit;
   unless ( $read == $limit ) {
      die "Can't read labels (read $read, expected $limit)";
   }
   my @btmp = unpack('C*', $labels);
   return \@btmp;
}

1;
