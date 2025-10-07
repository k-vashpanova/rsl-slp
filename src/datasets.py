class PoseDataset(Dataset):
    def __init__(self, pose_dir, use_existing=False, seed=None,
                 scaling=1, interpolation='linear',
                 reduce_upper_face_contours=True,
                 reduce_lower_face_contours=False,
                 reduce_eyes=False):

        self.pose_dir = pose_dir
        self.seed = seed
        self.gen = torch.Generator()

        self.interpolation = interpolation
        self.scaling = scaling

        self.reduce_upper_face_contours = reduce_upper_face_contours
        self.reduce_lower_face_contours = reduce_lower_face_contours
        self.reduce_eyes = reduce_eyes

        if use_existing:
            self.self_pose_dir = self.pose_dir
            self.poses_list = os.listdir(self.pose_dir)

        else:
            self.self_pose_dir = self.pose_dir + '_SignMAE'
            try:
                os.mkdir(self.self_pose_dir)
                print(f"Directory '{self.self_pose_dir}' created successfully.")
            except FileExistsError:
                print(f"Directory '{self.self_pose_dir}' already exists.")

            print('Preparing the dataset...')
            raw_poses_list = [p for p in os.listdir(self.pose_dir) if p[-5:] == '.pose']
            self.poses_list = [p for p in tqdm(raw_poses_list) if self.__valid__(p)]

    def __len__(self):
        return len(self.poses_list)

    def __getitem__(self, idx):
        pose_path = os.path.join(self.self_pose_dir, self.poses_list[idx])
        pose = self.get_pose(pose_path)
        return self.get_fragment(pose)

    def __valid__(self, pose_name):

        pose_path = os.path.join(self.pose_dir, pose_name)
        pose = self.get_pose(pose_path, preprocess=True)

        if (pose and pose.body.data.shape[0] > 15):
            with open(os.path.join(self.self_pose_dir, pose_name), "wb") as f:
                    pose.write(f)
            return 1

        else:
            return 0

    def filter_pose(self, pose):

        dists = [distance_batch(pose.body.data.data[i, 0], pose.body.data.data[i-1, 0]).mean()\
                 for i in range(1, pose.body.data.shape[0])]
        dists.insert(0, .0)

        mask = (np.array(dists) < 0.42)
        pose.body.data, pose.body.confidence = pose.body.data[mask], pose.body.confidence[mask]

        return pose

    def reduce_holistic(self, pose: Pose) -> Pose:
        known_pose_format = detect_known_pose_format(pose)
        if known_pose_format != "holistic":
            return pose

        # To avoid installing mediapipe, we just hardcode the face contours given the above code
        face_contours = [
            '0', '7', '10', '13', '14', '17', '21', '33', '37', '39', '40', '46', '52', '53', '54', '55', '58', '61', '63',
            '65', '66', '67', '70', '78', '80', '81', '82', '84', '87', '88', '91', '93', '95', '103', '105', '107', '109',
            '127', '132', '133', '136', '144', '145', '146', '148', '149', '150', '152', '153', '154', '155', '157', '158',
            '159', '160', '161', '162', '163', '172', '173', '176', '178', '181', '185', '191', '234', '246', '249', '251',
            '263', '267', '269', '270', '276', '282', '283', '284', '285', '288', '291', '293', '295', '296', '297', '300',
            '308', '310', '311', '312', '314', '317', '318', '321', '323', '324', '332', '334', '336', '338', '356', '361',
            '362', '365', '373', '374', '375', '377', '378', '379', '380', '381', '382', '384', '385', '386', '387', '388',
            '389', '390', '397', '398', '400', '402', '405', '409', '415', '454', '466'
        ]

        if self.reduce_upper_face_contours:
            upper_face = ['10', '21', '46', '52', '53', '54', '55', '65', '67', '103', '109',
                          '162', '251', '276', '282', '283', '284', '285', '295', '297', '332',
                          '338', '389']
            face_contours = [c for c in face_contours if c not in upper_face]

        if self.reduce_lower_face_contours:
            lower_face = ["127", "234", "93", "132", "58", "172", "136", "149", "152", "378",
                          "365", "397","288", "361", "323", "454", "356", "150", "176", "148",
                          "377", "400", "379"]
            face_contours = [c for c in face_contours if c not in lower_face]

        if self.reduce_eyes:
            eyes = ['246', '161', '159', '157', '173', '155', '154', '145', '163', '7',
                    '398', '384', '386', '388', '466', '249', '390', '374', '381', '382']
            face_contours = [c for c in face_contours if c not in eyes]

        ignore_names = [
            "EAR", "MOUTH", "EYE",  #Face
            "THUMB", "PINKY", "INDEX",  # Hands
            "KNEE", "ANKLE", "HEEL", "FOOT_INDEX"  # Feet
        ]

        body_component = [c for c in pose.header.components if c.name == "POSE_LANDMARKS"][0]
        body_no_face_no_hands = [p for p in body_component.points if all([i not in p for i in ignore_names])]

        components = [c.name for c in pose.header.components if c.name != "POSE_WORLD_LANDMARKS"]
        return pose.get_components(components, {"FACE_LANDMARKS": face_contours, "POSE_LANDMARKS": body_no_face_no_hands})

    def get_pose(self, path,
                 preprocess=False,
                 filter=True,
                 scaling=None,
                 interpolation=None):

        try:
            data_buffer = open(path, "rb").read()
            pose = Pose.read(data_buffer)

            if preprocess:

                if not interpolation:
                    interpolation = self.interpolation
                if not scaling:
                    scaling = self.scaling

                pose = correct_wrists(pose)
                pose = self.reduce_holistic(pose) #раньше был только он
                pose = pose_hide_legs(pose, remove=True)
                pose = pose.normalize(scale_factor=scaling)
                pose.focus()

                mask = ([bool(i.any()) for i in pose.body.data])
                pose.body.data, pose.body.confidence = pose.body.data[mask], pose.body.confidence[mask]

                if filter:
                    pose = self.filter_pose(pose)

                if interpolation != 'none' and pose.body.data.shape[0]>1:
                    pose.body = pose.body.interpolate(pose.body.fps, interpolation)

            return pose

        except KeyboardInterrupt:
            raise

        except:
            return None

    def get_fragment(self, pose):

        data = pose.body.torch().data.data #cordinates
        conf = pose.body.torch().confidence #confidence

        #fixed generation seed for test_dataset
        if self.seed is not None:
          self.gen = self.gen.manual_seed(self.seed)

        i = torch.randint(high=data.shape[0]-15, size=(1,), generator=self.gen)
        X = torch.concat((data, conf.unsqueeze(-1)), dim=-1)[i:i+15].squeeze()
        y = data[i:i+15].squeeze()

        return X.float(), y.float()


class CharTokenizer():
    def __init__(self, subword_prefix='##', special_tokens=['[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]'], lower=False):
        self.subword_prefix = subword_prefix
        self.special_tokens = special_tokens
        self.lower = lower

        self.chars = "!$%&'()*+,-./0123456789:;=?@ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz~ЁАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдежзийклмнопрстуфхцчшщъыьэюяё"
        self.pad = '[PAD]'
        self.unk = '[UNK]'
        self.sub = '[SUB]'
        if self.lower:
            self.chars = ''.join(set(self.chars.lower()))
            self.special_tokens = [t.lower() for t in self.special_tokens]
            self.subword_prefix = self.subword_prefix.lower()
        self.id2char = {i+3:char for i, char in enumerate(self.chars)}
        self.id2char.update(enumerate([self.pad, self.sub, self.unk]))
        self.char2id = {char:id for id, char in self.id2char.items()}
        self.max_length = 16

    def __call__(self, input, padding=False, truncation=False, max_length=None):

        if self.lower:
            output = [self.convert_tokens_to_ids(self.tokenize(word.lower())) for word in input]
        else:
            output = [self.convert_tokens_to_ids(self.tokenize(word)) for word in input]

        if not max_length:
            max_length = self.max_length
        if truncation:
            output = [word[:max_length] for word in output]
        if padding == 'max_length':
            output = [word+[self.char2id[self.pad]]*(max_length - len(word)) for word in output]
        elif padding == 'longest':
            longest = len(max([t for t in output], key=len))
            output = [word+[self.char2id[self.pad]]*(longest - len(word)) for word in output]

        return output

    def __len__(self):
        return len(self.id2char)

    def tokenize(self, text):
        if text in self.special_tokens:
            return [self.pad]
        elif text[:len(self.subword_prefix)] == self.subword_prefix:
            return [self.sub] + [*text[len(self.subword_prefix):]]
        else:
            return [*text]

    def convert_ids_to_tokens(self, ids):
        return [self.id2char.get(id) for id in ids]

    def convert_tokens_to_ids(self, tokens):
        return [self.char2id.get(token, self.char2id[self.unk]) for token in tokens]


class Text2PoseDataset():

    def __init__(self,
                 table_path,
                 pose_dir,
                 text_tokenizer,
                 split='train',
                 pose_window = 15,
                 max_length_pose_tokens=32,
                 max_length_text_tokens=64,
                 max_length_char_tokens=16):
        super().__init__()

        self.pose_window = pose_window
        self.max_length_pose_tokens=max_length_pose_tokens
        self.max_length_text_tokens=max_length_text_tokens
        self.max_length_char_tokens=max_length_char_tokens
        self.pose_dir = pose_dir
        self.split=split

        ext = table_path.split('.')[-1]
        if ext == 'csv':
            table = pd.read_csv(table_path, index_col=0)
        elif ext in ['xls', 'xlsx']:
            table = pd.read_excel(table_path, index_col=0)
        else:
            raise

        self.p2t = {i.pose:i.text for i in table.itertuples() \
                    if i.pose in os.listdir(self.pose_dir) \
                    and i.split==self.split}
        self.poses_list = list(self.p2t.keys())
        self.text_tokenizer = text_tokenizer
        self.char_tokenizer = CharTokenizer(lower=True, special_tokens=self.text_tokenizer.all_special_tokens)

    def __len__(self):
        return len(self.poses_list)

    def __getitem__(self, idx):
        pose_name = self.poses_list[idx]
        frames, length_mask = self.tokenize_pose(pose_name)
        tokenized, chars, text = self.tokenize_text(self.p2t[pose_name])
        return tokenized.input_ids, tokenized.attention_mask, chars, text, frames, length_mask

    def tokenize_pose(self, pose):
        frames = self.get_pose(pose)
        frames = frames[:self.max_length_pose_tokens * self.pose_window]
        L, V, C = frames.shape
        pad_frame = frames[-1:]
        padded_frames = pad_frame.repeat(repeats=self.max_length_pose_tokens * self.pose_window, axis=0)
        padded_frames[:L] = frames

        length_mask = np.zeros((self.max_length_pose_tokens))
        length_mask[:(L/self.pose_window).__ceil__()] = 1.

        return padded_frames, length_mask

    def tokenize_text(self, text):
        tokenized = self.text_tokenizer(text, truncation=True, max_length=self.max_length_text_tokens, padding='max_length')
        chars = self.char_tokenizer(self.text_tokenizer.convert_ids_to_tokens(tokenized.input_ids),
                                    truncation=True, max_length=self.max_length_char_tokens, padding='max_length')
        return tokenized, chars, text

    def get_pose(self, file_name):
        path = os.path.join(self.pose_dir, file_name)
        data_buffer = open(path, "rb").read()
        pose = Pose.read(data_buffer)
        pose = pose.body.data.data.squeeze()
        return pose

    def collate_fn(self, batch):

        input_ids, attention_mask, chars, text, frames, length_mask = zip(*batch)

        X = {
            'input_ids': torch.tensor(input_ids).to(DEVICE),
            'attention_mask': torch.tensor(attention_mask).to(DEVICE),
            'chars': torch.tensor(chars).to(DEVICE)
        }

        y = {
            'text': text,
            'frames': torch.tensor(np.array(frames)).float().to(DEVICE),
            'length_mask': torch.tensor(np.array(length_mask)).float().to(DEVICE),
        }

        return X, y
