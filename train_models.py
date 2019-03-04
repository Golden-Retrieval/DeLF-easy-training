def train_resnet(config):
    sess = config.sess
    #################[ Image configure ]##################
    conv_epoch = config.nb_epoch - config.fc_epoch
    images_holder = config.images_holder
    labels_holder = tf.placeholder(shape=(None,), dtype=tf.int32)
    fc_train = True

    t0 = time.time()

    ############## [ loss, optimizer, variable ] ##############
    tf.losses.sparse_softmax_cross_entropy(labels=labels_holder,
                                           logits=config.logits)
    loss = tf.losses.get_total_loss()

    ######### [ Convolution Layer Optimizer + FC Layer Optimizer] ########
    conv_optimizer = tf.train.AdamOptimizer(
        learning_rate=config.conv_learning_rate)
    conv_train_variables = [v for v in tf.global_variables()
                            if ('Adam' not in v.name)]
    conv_grad_update = conv_optimizer.minimize(loss,
                                               var_list=conv_train_variables)
    fc_optimizer = tf.train.AdamOptimizer(
        learning_rate=config.fc_learning_rate)
    fc_train_variables = [v for v in tf.global_variables()
                          if ('resnet' not in v.name) and (
                              'Adam' not in v.name)]
    fc_grad_update = fc_optimizer.minimize(loss,
                                           var_list=fc_train_variables)

    ############## [ prediction, accuracy, init op ] ##############
    prediction = tf.to_int32(tf.argmax(config.logits, 1))
    correct_prediction = tf.equal(prediction, labels_holder)
    accuracy = tf.reduce_mean(
        tf.cast(correct_prediction, tf.float32))

    # global initializer
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    ############## [ restore variable from exist checkpoint] ##############
    restore_var = [v for v in tf.global_variables() if
                   ('resnet' in v.name) and ('Adam' not in v.name)]
    saver = tf.train.Saver(restore_var, max_to_keep=5)
    saver.restore(sess, config.restore_file)
    print("weights loaded. ")
    print("there are " + str(config.num_train_batches) + " batches")

    ############## [ FC training + Conv training] ##############
    for epoch in range(config.nb_epoch):
        if (epoch >= config.fc_epoch):
            fc_train = False

        if (fc_train):
            sess.run(tf.variables_initializer(
                fc_optimizer.variables()))  # adam optimizer 때문에 추가
            grad_update = fc_grad_update
        else:
            sess.run(tf.variables_initializer(
                conv_optimizer.variables()))  # adam optimizer 때문에 추가
            grad_update = conv_grad_update

        print(
            'Starting epoch %d / %d' % (epoch + 1, config.nb_epoch))
        t1 = time.time()
        loss_sum = 0.0
        acc_sum = 0.0

        # ==========[ training ]==========
        sess.run(config.train_init_op)
        for batch in range(config.num_train_batches):
            input_images, input_labels = sess.run(
                [config.images, config.labels])
            print(input_images.shape)
            print(input_labels.shape)
            feed_dict = {images_holder: input_images,
                         labels_holder: input_labels}
            _, batch_acc, batch_loss = sess.run(
                [grad_update, accuracy, loss], feed_dict=feed_dict)
            acc_sum += batch_acc.item()
            loss_sum += batch_loss.item()
            postfix = "batch_acc : %.6f, batch_loss : %.6f" % (
                batch_acc, batch_loss)
            print("%d/%d " % (
                batch, config.num_train_batches) + postfix)

        # training set의 평균 accuracy, loss
        avg_acc = acc_sum / config.num_train_batches
        avg_loss = loss_sum / config.num_train_batches

        # =========[ validation ]==========
        val_acc = 0.0
        sess.run(config.validation_init_op)
        for batch in range(config.num_val_batches):
            input_images, input_labels = sess.run(
                [config.images, config.labels])
            feed_dict = {images_holder: input_images,
                         labels_holder: input_labels}
            batch_acc, batch_loss = sess.run([accuracy, loss],
                                             feed_dict=feed_dict)
            val_acc += batch_acc.item()

        # validation set의 평균 accuracy
        val_acc = val_acc / config.num_val_batches

        # =======[ training 결과출력 ]=======
        print("=" * config.dash_size)
        print(
            'epoch: %d, train_loss: %.6f, train_acc: %.6f, val_acc: %6f' % (
                epoch, avg_loss, avg_acc, val_acc))
        print("=" * config.dash_size)
        t2 = time.time()
        print('Training time for one epoch : %.1f' % ((t2 - t1)))

        # ========[ 현재 epoch 저장 ]========
        save_time = time.strftime('+%Y-%m-%d_%H:%M',
                                  time.localtime(time.time()))
        save_name_time = config.save_name + "_%d" % (
            epoch) + save_time
        saver.save(sess, save_name_time)
    # 끝
    print('\nTotal training time : %.1f' % (time.time() - t0))
    sess.close()