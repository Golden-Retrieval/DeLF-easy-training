import tensorflow as tf 
import time 


def train_model(config):
    sess = config.sess
    saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=2)
    
    ############## [ Image configure ] ########################
    conv_epoch = config.nb_epoch - config.fc_epoch
    images_holder = config.images_holder
    labels_holder = tf.placeholder(shape=(None,), dtype=tf.int32)
    fc_train = True
    t0 = time.time()

    ############## [ loss, optimizer, variable ] ##############
    tf.losses.sparse_softmax_cross_entropy(labels=labels_holder,
                                           logits=config.logits)
    loss = tf.losses.get_total_loss()

    ##### [ Convolution Layer Optimizer + FC Layer Optimizer] ##
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

    ##################### [ Att Optimizer] #####################
    att_optimizer = tf.train.AdamOptimizer(
        learning_rate=config.att_learning_rate)
    att_train_variables = [v for v in tf.global_variables()
                           if
                           ('Adam' not in v.name) and ('resnet' not in v.name)]
    att_grad_update = att_optimizer.minimize(loss,
                                             var_list=att_train_variables)

    ############## [ Adam Variables Init ] #####################
    sess.run(tf.variables_initializer(fc_optimizer.variables()))
    sess.run(tf.variables_initializer(conv_optimizer.variables()))
    sess.run(tf.variables_initializer(att_optimizer.variables()))

    ############## [ prediction, accuracy, init op ] ##############
    prediction = tf.to_int32(tf.argmax(config.logits, 1))
    correct_prediction = tf.equal(prediction, labels_holder)
    accuracy = tf.reduce_mean(
        tf.cast(correct_prediction, tf.float32))

    ############## [ FC training + Conv training] ##############
    print("= batches =: " + str(config.num_train_batches))
    for epoch in range(1, config.nb_epoch + 1):

        # ========== [ select train mode ] ==============
        if config.train_step == "resnet_finetune":
            if (epoch > config.fc_epoch):
                grad_update = conv_grad_update
            else:
                grad_update = fc_grad_update
        elif config.train_step == "att_learning":
            grad_update = att_grad_update
        else:
            raise Exception("Invalid config.train_step")

        print('Starting epoch %d / %d' % (epoch, config.nb_epoch))
        t1 = time.time()
        loss_sum = 0.0
        acc_sum = 0.0

        # ============== [ training ] ==============
        sess.run(config.train_init_op)
        for batch in range(config.num_train_batches):
            input_images, input_labels = sess.run(
                [config.images, config.labels])
            feed_dict = {images_holder: input_images,
                         labels_holder: input_labels}
            _, batch_acc, batch_loss = sess.run(
                [grad_update, accuracy, loss], feed_dict=feed_dict)
            acc_sum += batch_acc.item()
            loss_sum += batch_loss.item()
            if batch % 10 == 0:
                postfix = "batch_acc : %.6f, batch_loss : %.6f" % (
                    batch_acc, batch_loss)
                print("%d/%d " % (
                    batch, config.num_train_batches) + postfix)

        # training set의 평균 accuracy, loss
        avg_acc = acc_sum / config.num_train_batches
        avg_loss = loss_sum / config.num_train_batches

        # ============== [ validation ] ==============
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

        # ============== [ training 결과출력 ] ==============
        print("=" * config.dash_size)
        print(
            'epoch: %d, train_loss: %.6f, train_acc: %.6f, val_acc: %6f' % (
                epoch, avg_loss, avg_acc, val_acc))
        print("=" * config.dash_size)
        t2 = time.time()
        print('Training time for one epoch : %.1f' % ((t2 - t1)))

        # ============== [ 현재 epoch 저장 ] ==============
        save_name_epoch = config.save_name + "_%d" % (
            epoch)
        saver.save(sess, save_name_epoch)
    print('\nTotal training time : %.1f' % (time.time() - t0))
    sess.close()
