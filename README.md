# Play tic-tac-toe with OpenCV
This project is about the traditional game of tic-tac-toe. Using OpenCV and Keras we are now able to play it
against the computer using a real board and a marker.

## Requirements
- Sheet of paper
- Black marker
- USB camera to stream video feed directly to the computer

## Guide
Everything you need to know about how to play is described in the video below.

**Insert video**

## How to win?
Since the computer uses the [alphabeta algorithm](https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning), it is
guaranteed the game will end up being a draw or a win for the computer. You can see how the algorithm works by running
the *alphabeta.py* script. Have fun!

## Inner workings of the classification model
I used a small Convolutional Neural Network to classify for each cell in the grid (board) whether there's an X, an O
or nothing. It has an average score of 96.91%, but it has worked well everytime I've played. Specifics 
of the model and it's configuration are in the [README.md](data/README.md) under the data directory.

## Resources
I used [CWoebker's alphabeta script](https://cwoebker.com/posts/tic-tac-toe) and added some documentation to it.
