"""Game of tic tac toe using OpenCV to play against computer"""


import os
import sys
import cv2
import argparse
import numpy as np

from keras.models import load_model

from utils import imutils
from utils import detections
from alphabeta import Tic, get_enemy, determine


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('cam', type=int,
                        help='USB camera for video streaming')
    parser.add_argument('--model', '-m', type=str, default='data/model.h5',
                        help='model file (.h5) to detect Xs and Os')

    return parser.parse_args()


def find_sheet_paper(frame, thresh, add_margin=True):
    """Detect the coords of the sheet of paper the game will be played on"""
    stats = detections.find_corners(thresh)
    # First point is center of coordinate system, so ignore it
    # We only want sheet of paper's corners
    corners = stats[1:, :2]
    corners = imutils.order_points(corners)
    # Get bird view of sheet of paper
    paper = imutils.four_point_transform(frame, corners)
    if add_margin:
        paper = paper[10:-10, 10:-10]
    return paper, corners


def find_shape(cell):
    """Is shape and X or an O?"""
    mapper = {0: None, 1: 'X', 2: 'O'}
    cell = detections.preprocess_input(cell)
    idx = np.argmax(model.predict(cell))
    return mapper[idx]


def get_board_template(thresh):
    """Returns 3 x 3 grid, a.k.a the board"""
    # Find grid's center cell, and based on it fetch
    # the other eight cells
    middle_center = detections.contoured_bbox(thresh)
    center_x, center_y, width, height = middle_center

    # Useful coords
    left = center_x - width
    right = center_x + width
    top = center_y - height
    bottom = center_y + height

    # Middle row
    middle_left = (left, center_y, width, height)
    middle_right = (right, center_y, width, height)
    # Top row
    top_left = (left, top, width, height)
    top_center = (center_x, top, width, height)
    top_right = (right, top, width, height)
    # Bottom row
    bottom_left = (left, bottom, width, height)
    bottom_center = (center_x, bottom, width, height)
    bottom_right = (right, bottom, width, height)

    # Grid's coordinates
    return [top_left, top_center, top_right,
            middle_left, middle_center, middle_right,
            bottom_left, bottom_center, bottom_right]


def draw_shape(template, shape, coords):
    """Draw on a cell the shape which resides in it"""
    x, y, w, h = coords
    if shape == 'O':
        centroid = (x + int(w / 2), y + int(h / 2))
        cv2.circle(template, centroid, 10, (0, 0, 0), 2)
    elif shape == 'X':
        # Draws the 'X' shape
        cv2.line(template, (x + 10, y + 7), (x + w - 10, y + h - 7),
                 (0, 0, 0), 2)
        cv2.line(template, (x + 10, y + h - 7), (x + w - 10, y + 7),
                 (0, 0, 0), 2)
    return template


def play(vcap):
    """Play tic tac toe game with computer that uses the alphabeta algorithm"""
    # Initialize opponent (computer)
    board = Tic()
    history = {}
    message = True
    # Start playing
    while True:
        ret, frame = vcap.read()
        key = cv2.waitKey(1) & 0xFF
        if not ret:
            print('[INFO] finished video processing')
            break

        # Stop
        if key == ord('q'):
            print('[INFO] stopped video processing')
            break

        # Preprocess input
        # frame = imutils.resize(frame, 500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)
        thresh = cv2.GaussianBlur(thresh, (7, 7), 0)
        paper, corners = find_sheet_paper(frame, thresh)
        # Four red dots must appear on each corner of the sheet of paper,
        # otherwise try moving it until they're well detected
        for c in corners:
            cv2.circle(frame, tuple(c), 2, (0, 0, 255), 2)

        # Now working with 'paper' to find grid
        paper_gray = cv2.cvtColor(paper, cv2.COLOR_BGR2GRAY)
        _, paper_thresh = cv2.threshold(
            paper_gray, 170, 255, cv2.THRESH_BINARY_INV)
        grid = get_board_template(paper_thresh)

        # Draw grid and wait until user makes a move
        for i, (x, y, w, h) in enumerate(grid):
            cv2.rectangle(paper, (x, y), (x + w, y + h), (0, 0, 0), 2)
            if history.get(i) is not None:
                shape = history[i]['shape']
                paper = draw_shape(paper, shape, (x, y, w, h))

        # Make move
        if message:
            print('Make move, then press spacebar')
            message = False
        if not key == 32:
            cv2.imshow('original', frame)
            cv2.imshow('bird view', paper)
            continue
        player = 'X'

        # User's time to play, detect for each available cell
        # where has he played
        available_moves = np.delete(np.arange(9), list(history.keys()))
        for i, (x, y, w, h) in enumerate(grid):
            if i not in available_moves:
                continue
            # Find what is inside each free cell
            cell = paper_thresh[int(y): int(y + h), int(x): int(x + w)]
            shape = find_shape(cell)
            if shape is not None:
                history[i] = {'shape': shape, 'bbox': (x, y, w, h)}
                board.make_move(i, player)
            paper = draw_shape(paper, shape, (x, y, w, h))

        # Check whether game has finished
        if board.complete():
            break

        # Computer's time to play
        player = get_enemy(player)
        computer_move = determine(board, player)
        board.make_move(computer_move, player)
        history[computer_move] = {'shape': 'O', 'bbox': grid[computer_move]}
        paper = draw_shape(paper, 'O', grid[computer_move])

        # Check whether game has finished
        if board.complete():
            break

        # Show images
        cv2.imshow('original', frame)
        # cv2.imshow('thresh', paper_thresh)
        cv2.imshow('bird view', paper)
        message = True

    # Show winner
    winner = board.winner()
    height = paper.shape[0]
    text = 'Winner is {}'.format(str(winner))
    cv2.putText(paper, text, (10, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imshow('bird view', paper)
    cv2.waitKey(0) & 0xFF

    # Close windows
    vcap.release()
    cv2.destroyAllWindows()
    return board.winner()


def main(args):
    """Check if everything's okay and start game"""
    # Load model
    global model
    assert os.path.exists(args.model), '{} does not exist'
    model = load_model(args.model)

    # Initialize webcam feed
    vcap = cv2.VideoCapture(args.cam)
    if not vcap.isOpened():
        raise IOError('could not get feed from cam #{}'.format(args.cam))

    # Announce winner!
    winner = play(vcap)
    print('Winner is:', winner)
    sys.exit()


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
